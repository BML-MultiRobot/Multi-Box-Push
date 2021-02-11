// Auto-generated. Do not edit!

// (in-package multi_cla.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class Location {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.x_global = null;
      this.y_global = null;
    }
    else {
      if (initObj.hasOwnProperty('x_global')) {
        this.x_global = initObj.x_global
      }
      else {
        this.x_global = 0.0;
      }
      if (initObj.hasOwnProperty('y_global')) {
        this.y_global = initObj.y_global
      }
      else {
        this.y_global = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Location
    // Serialize message field [x_global]
    bufferOffset = _serializer.float32(obj.x_global, buffer, bufferOffset);
    // Serialize message field [y_global]
    bufferOffset = _serializer.float32(obj.y_global, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Location
    let len;
    let data = new Location(null);
    // Deserialize message field [x_global]
    data.x_global = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [y_global]
    data.y_global = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a message object
    return 'multi_cla/Location';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '7786b133d00488af6ca566640ecc3e11';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 x_global
    float32 y_global
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Location(null);
    if (msg.x_global !== undefined) {
      resolved.x_global = msg.x_global;
    }
    else {
      resolved.x_global = 0.0
    }

    if (msg.y_global !== undefined) {
      resolved.y_global = msg.y_global;
    }
    else {
      resolved.y_global = 0.0
    }

    return resolved;
    }
};

module.exports = Location;
