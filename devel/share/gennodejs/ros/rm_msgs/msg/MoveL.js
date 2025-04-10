// Auto-generated. Do not edit!

// (in-package rm_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class MoveL {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.Pose = null;
      this.speed = null;
      this.trajectory_connect = null;
    }
    else {
      if (initObj.hasOwnProperty('Pose')) {
        this.Pose = initObj.Pose
      }
      else {
        this.Pose = new geometry_msgs.msg.Pose();
      }
      if (initObj.hasOwnProperty('speed')) {
        this.speed = initObj.speed
      }
      else {
        this.speed = 0.0;
      }
      if (initObj.hasOwnProperty('trajectory_connect')) {
        this.trajectory_connect = initObj.trajectory_connect
      }
      else {
        this.trajectory_connect = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MoveL
    // Serialize message field [Pose]
    bufferOffset = geometry_msgs.msg.Pose.serialize(obj.Pose, buffer, bufferOffset);
    // Serialize message field [speed]
    bufferOffset = _serializer.float32(obj.speed, buffer, bufferOffset);
    // Serialize message field [trajectory_connect]
    bufferOffset = _serializer.uint8(obj.trajectory_connect, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MoveL
    let len;
    let data = new MoveL(null);
    // Deserialize message field [Pose]
    data.Pose = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [speed]
    data.speed = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [trajectory_connect]
    data.trajectory_connect = _deserializer.uint8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 61;
  }

  static datatype() {
    // Returns string type for a message object
    return 'rm_msgs/MoveL';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '71f8a77c6be4dc679da6e3cd77730408';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Pose Pose
    float32 speed
    uint8 trajectory_connect
    ================================================================================
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MoveL(null);
    if (msg.Pose !== undefined) {
      resolved.Pose = geometry_msgs.msg.Pose.Resolve(msg.Pose)
    }
    else {
      resolved.Pose = new geometry_msgs.msg.Pose()
    }

    if (msg.speed !== undefined) {
      resolved.speed = msg.speed;
    }
    else {
      resolved.speed = 0.0
    }

    if (msg.trajectory_connect !== undefined) {
      resolved.trajectory_connect = msg.trajectory_connect;
    }
    else {
      resolved.trajectory_connect = 0
    }

    return resolved;
    }
};

module.exports = MoveL;
