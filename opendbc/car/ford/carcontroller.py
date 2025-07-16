import math
import numpy as np
from opendbc.can.packer import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_std_steer_angle_limits, structs
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags
from opendbc.car.interfaces import CarControllerBase, ISO_LATERAL_ACCEL, V_CRUISE_MAX
from opendbc.car.carlog import carlog
from common.params import Params
from common.pid import PIDController

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

# CAN FD limits:
# Limit to average banked road since safety doesn't have the roll
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation. higher actual roll raises lateral acceleration
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~2.4 m/s^2


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle, lat_active, CP):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = np.clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                              current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, steering_angle, lat_active, CarControllerParams.ANGLE_LIMITS)

  # Ford Q4/CAN FD has more torque available compared to Q3/CAN so we limit it based on lateral acceleration.
  # Safety is not aware of the road roll so we subtract a conservative amount at all times
  if CP.flags & FordFlags.CANFD:
    # Limit curvature to conservative max lateral acceleration
    curvature_accel_limit = MAX_LATERAL_ACCEL / (max(v_ego_raw, 1) ** 2)
    apply_curvature = float(np.clip(apply_curvature, -curvature_accel_limit, curvature_accel_limit))

  return apply_curvature


def apply_creep_compensation(accel: float, v_ego: float) -> float:
  creep_accel = np.interp(v_ego, [1., 3.], [0.6, 0.])
  creep_accel = np.interp(accel, [0., 0.2], [creep_accel, 0.])
  accel -= creep_accel
  return float(accel)


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.CAN = fordcan.CanBus(CP)
    self.params = Params()
    self.pid = PIDController(k_p=CarControllerParams.APA_PID_GAINS[0],
                           k_i=CarControllerParams.APA_PID_GAINS[1],
                           k_d=CarControllerParams.APA_PID_GAINS[2],
                           k_f=0.,  # 前馈项在这里不使用
                           pos_limit=self.CP.steerAngleMax,
                           neg_limit=-self.CP.steerAngleMax,
                           anti_windup_rate=CarControllerParams.APA_PID_WINDUP_LIMIT)

    self.apply_curvature_last = 0
    self.last_steering_angle_deg = 0.
    self.ping_pong_state = 1
    self.ping_pong_frame_counter = 0
    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      # 如果APA转向辅助已启用，则使用新的APA命令
      if self.params.get_bool("DynamicSteeringEnabled"):
        # 0. 检查PSCM状态，决定是否可以发送控制指令
        # 驾驶员介入或永久故障时，重置并禁用
        if CS.out.steerFaultDriverOverride or CS.out.steerFaultPermanent:
          self.pid.reset()
          steering_angle_deg = CS.out.steeringAngleDeg # 使用当前角度作为输出，实现平滑退出
          self.last_steering_angle_deg = steering_angle_deg
          can_sends.append(fordcan.create_apa_steering_command(self.packer, self.CAN, steering_angle_deg, CS.out.gearShifter))
        else:
          # 1. 计算期望转向角度 (Setpoint)
          desired_angle_deg = -actuators.curvature * self.CP.steerRatio * 180 / 3.14159

          # 2. 获取当前实际转向角度 (Measurement)
          current_angle_deg = CS.out.steeringAngleDeg

          # 3. 智能 "乒乓模式" 和 PID 积分项管理
          # 当PSCM明确报告无法达到角度时，激活乒乓模式并冻结积分项
          is_angle_not_reached = CS.out.steerFaultAngleNotReached
          self.pid.freeze_I = is_angle_not_reached

          if is_angle_not_reached:
            ping_pong_period_frames = (1.0 / CarControllerParams.PING_PONG_FREQUENCY) * (1.0 / (CarControllerParams.STEER_STEP * DT_CTRL)) / 2.0
            self.ping_pong_frame_counter += 1
            if self.ping_pong_frame_counter > ping_pong_period_frames:
              self.ping_pong_state *= -1
              self.ping_pong_frame_counter = 0
            desired_angle_deg += self.ping_pong_state * CarControllerParams.PING_PONG_AMPLITUDE
          else:
            self.ping_pong_frame_counter = 0
            self.ping_pong_state = 1

          # 4. 更新PID控制器
          steering_angle_deg = self.pid.update(desired_angle_deg, current_angle_deg,
                                               speed=CS.out.vEgo,
                                               feedforward=desired_angle_deg)

          # 5. 应用转向速率限制，平滑最终的转向指令
          steering_angle_deg = np.clip(steering_angle_deg,
                                       self.last_steering_angle_deg - CarControllerParams.STEER_ANGLE_RATE_LIMIT,
                                       self.last_steering_angle_deg + CarControllerParams.STEER_ANGLE_RATE_LIMIT)
          self.last_steering_angle_deg = steering_angle_deg

          # 添加日志
          if self.frame % 100 == 0:
            carlog.debug(f"APA PID: Desired={desired_angle_deg:.2f}, Current={current_angle_deg:.2f}, Final={steering_angle_deg:.2f}, APA_Status={CS.out.apaLaneAssistStatus}")

          # 6. 发送最终命令
          can_sends.append(fordcan.create_apa_steering_command(self.packer, self.CAN, steering_angle_deg, CS.out.gearShifter))
      else:
        # 重置PID积分项，防止在功能关闭时累积误差
        self.pid.reset()

        # apply rate limits, curvature error limit, and clip to signal range
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        self.apply_curvature_last = apply_ford_curvature_limits(actuators.curvature, self.apply_curvature_last, current_curvature,
                                                                CS.out.vEgoRaw, 0., CC.latActive, self.CP)

        if self.CP.flags & FordFlags.CANFD:
          # TODO: extended mode
          # Ford uses four individual signals to dictate how to drive to the car. Curvature alone (limited to 0.02m/s^2)
          # can actuate the steering for a large portion of any lateral movements. However, in order to get further control on
          # steer actuation, the other three signals are necessary. Ford controls vehicles differently than most other makes.
          # A detailed explanation on ford control can be found here:
          # https://www.f150gen14.com/forum/threads/introducing-bluepilot-a-ford-specific-fork-for-comma3x-openpilot.24241/#post-457706
          mode = 1 if CC.latActive else 0
          counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
          can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode, 0., 0., -self.apply_curvature_last, 0., counter))
        else:
          can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive, 0., 0., -self.apply_curvature_last, 0.))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      accel = actuators.accel
      gas = accel

      if CC.longActive:
        # Compensate for engine creep at low speed.
        # Either the ABS does not account for engine creep, or the correction is very slow
        # TODO: verify this applies to EV/hybrid
        accel = apply_creep_compensation(accel, CS.out.vEgo)

        # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
        # however even 3.5 m/s^3 causes some overshoot with a step response.
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      gas = float(np.clip(gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      # Both gas and accel are in m/s^2, accel is used solely for braking
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      # PCM applies pitch compensation to gas/accel, but we need to compensate for the brake/pre-charge bits
      accel_due_to_pitch = 0.0
      if len(CC.orientationNED) == 3:
        accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

      accel_pitch_compensated = accel + accel_due_to_pitch

      # 创建一个"滑行死区"，以优先使用发动机制动而非机械刹车
      # 仅当需要的减速度足够大时，才请求刹车
      if not CC.longActive:
        # 如果ACC关闭，则永不请求刹车
        self.brake_request = False
      elif accel_pitch_compensated > CarControllerParams.ZERO_GAS:
        # 如果目标加速度为正，明确不请求刹车
        self.brake_request = False
      elif accel_pitch_compensated < CarControllerParams.BRAKE_PRESSED_GAS:
        # 如果目标加速度小于一个明确的负阈值，明确请求刹车
        self.brake_request = True
      # 在 ZERO_GAS 和 BRAKE_PRESSED_GAS 之间的区域，保持上一个刹车状态
      # 这形成了一个滞后区间，防止刹车在临界点抖动

      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      # TODO: look into using the actuators packet to send the desired speed
      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping, self.brake_request, v_ego_kph=V_CRUISE_MAX))

      self.accel = accel
      self.gas = gas

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

    # send acc ui msg at 5Hz or if ui state changes
    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      send_ui = True
      self.distance_bar_frame = self.frame

    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      show_distance_bars = self.frame - self.distance_bar_frame < 400
      can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                                 fcw_alert, CS.out.cruiseState.standstill, show_distance_bars,
                                                 hud_control, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars

    new_actuators = actuators.as_builder()
    new_actuators.curvature = self.apply_curvature_last
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
    return new_actuators, can_sends
