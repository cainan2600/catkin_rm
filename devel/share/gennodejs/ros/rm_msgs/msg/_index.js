
"use strict";

let Force_Position_Move_Joint = require('./Force_Position_Move_Joint.js');
let Pos_Teach = require('./Pos_Teach.js');
let ChangeTool_Name = require('./ChangeTool_Name.js');
let write_single_register = require('./write_single_register.js');
let MoveJ_P = require('./MoveJ_P.js');
let Arm_Joint_Speed_Max = require('./Arm_Joint_Speed_Max.js');
let Servo_GetAngle = require('./Servo_GetAngle.js');
let Hand_Posture = require('./Hand_Posture.js');
let Arm_Digital_Output = require('./Arm_Digital_Output.js');
let Joint_Current = require('./Joint_Current.js');
let Six_Force = require('./Six_Force.js');
let Gripper_Set = require('./Gripper_Set.js');
let Joint_Step = require('./Joint_Step.js');
let Joint_Enable = require('./Joint_Enable.js');
let Stop = require('./Stop.js');
let Tool_Analog_Output = require('./Tool_Analog_Output.js');
let Force_Position_State = require('./Force_Position_State.js');
let MoveL = require('./MoveL.js');
let JointPos = require('./JointPos.js');
let Hand_Speed = require('./Hand_Speed.js');
let Ort_Teach = require('./Ort_Teach.js');
let Gripper_Pick = require('./Gripper_Pick.js');
let Tool_Digital_Output = require('./Tool_Digital_Output.js');
let MoveC = require('./MoveC.js');
let Lift_Height = require('./Lift_Height.js');
let IO_Update = require('./IO_Update.js');
let Manual_Set_Force_Pose = require('./Manual_Set_Force_Pose.js');
let Set_Realtime_Push = require('./Set_Realtime_Push.js');
let Tool_IO_State = require('./Tool_IO_State.js');
let Socket_Command = require('./Socket_Command.js');
let ChangeWorkFrame_Name = require('./ChangeWorkFrame_Name.js');
let Start_Multi_Drag_Teach = require('./Start_Multi_Drag_Teach.js');
let Cabinet = require('./Cabinet.js');
let write_register = require('./write_register.js');
let Lift_Speed = require('./Lift_Speed.js');
let CartePos = require('./CartePos.js');
let LiftState = require('./LiftState.js');
let set_modbus_mode = require('./set_modbus_mode.js');
let Hand_Angle = require('./Hand_Angle.js');
let Arm_Analog_Output = require('./Arm_Analog_Output.js');
let Stop_Teach = require('./Stop_Teach.js');
let Arm_IO_State = require('./Arm_IO_State.js');
let MoveJ = require('./MoveJ.js');
let Joint_Teach = require('./Joint_Teach.js');
let Joint_Error_Code = require('./Joint_Error_Code.js');
let Plan_State = require('./Plan_State.js');
let Turtle_Driver = require('./Turtle_Driver.js');
let Set_Force_Position = require('./Set_Force_Position.js');
let Arm_Current_State = require('./Arm_Current_State.js');
let Force_Position_Move_Pose = require('./Force_Position_Move_Pose.js');
let Hand_Force = require('./Hand_Force.js');
let ChangeTool_State = require('./ChangeTool_State.js');
let Joint_Max_Speed = require('./Joint_Max_Speed.js');
let ArmState = require('./ArmState.js');
let ChangeWorkFrame_State = require('./ChangeWorkFrame_State.js');
let Servo_Move = require('./Servo_Move.js');
let GetArmState_Command = require('./GetArmState_Command.js');
let Hand_Seq = require('./Hand_Seq.js');

module.exports = {
  Force_Position_Move_Joint: Force_Position_Move_Joint,
  Pos_Teach: Pos_Teach,
  ChangeTool_Name: ChangeTool_Name,
  write_single_register: write_single_register,
  MoveJ_P: MoveJ_P,
  Arm_Joint_Speed_Max: Arm_Joint_Speed_Max,
  Servo_GetAngle: Servo_GetAngle,
  Hand_Posture: Hand_Posture,
  Arm_Digital_Output: Arm_Digital_Output,
  Joint_Current: Joint_Current,
  Six_Force: Six_Force,
  Gripper_Set: Gripper_Set,
  Joint_Step: Joint_Step,
  Joint_Enable: Joint_Enable,
  Stop: Stop,
  Tool_Analog_Output: Tool_Analog_Output,
  Force_Position_State: Force_Position_State,
  MoveL: MoveL,
  JointPos: JointPos,
  Hand_Speed: Hand_Speed,
  Ort_Teach: Ort_Teach,
  Gripper_Pick: Gripper_Pick,
  Tool_Digital_Output: Tool_Digital_Output,
  MoveC: MoveC,
  Lift_Height: Lift_Height,
  IO_Update: IO_Update,
  Manual_Set_Force_Pose: Manual_Set_Force_Pose,
  Set_Realtime_Push: Set_Realtime_Push,
  Tool_IO_State: Tool_IO_State,
  Socket_Command: Socket_Command,
  ChangeWorkFrame_Name: ChangeWorkFrame_Name,
  Start_Multi_Drag_Teach: Start_Multi_Drag_Teach,
  Cabinet: Cabinet,
  write_register: write_register,
  Lift_Speed: Lift_Speed,
  CartePos: CartePos,
  LiftState: LiftState,
  set_modbus_mode: set_modbus_mode,
  Hand_Angle: Hand_Angle,
  Arm_Analog_Output: Arm_Analog_Output,
  Stop_Teach: Stop_Teach,
  Arm_IO_State: Arm_IO_State,
  MoveJ: MoveJ,
  Joint_Teach: Joint_Teach,
  Joint_Error_Code: Joint_Error_Code,
  Plan_State: Plan_State,
  Turtle_Driver: Turtle_Driver,
  Set_Force_Position: Set_Force_Position,
  Arm_Current_State: Arm_Current_State,
  Force_Position_Move_Pose: Force_Position_Move_Pose,
  Hand_Force: Hand_Force,
  ChangeTool_State: ChangeTool_State,
  Joint_Max_Speed: Joint_Max_Speed,
  ArmState: ArmState,
  ChangeWorkFrame_State: ChangeWorkFrame_State,
  Servo_Move: Servo_Move,
  GetArmState_Command: GetArmState_Command,
  Hand_Seq: Hand_Seq,
};
