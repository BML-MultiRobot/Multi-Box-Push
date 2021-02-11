; Auto-generated. Do not edit!


(cl:in-package multi_cla-msg)


;//! \htmlinclude Location.msg.html

(cl:defclass <Location> (roslisp-msg-protocol:ros-message)
  ((x_global
    :reader x_global
    :initarg :x_global
    :type cl:float
    :initform 0.0)
   (y_global
    :reader y_global
    :initarg :y_global
    :type cl:float
    :initform 0.0))
)

(cl:defclass Location (<Location>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Location>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Location)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name multi_cla-msg:<Location> is deprecated: use multi_cla-msg:Location instead.")))

(cl:ensure-generic-function 'x_global-val :lambda-list '(m))
(cl:defmethod x_global-val ((m <Location>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader multi_cla-msg:x_global-val is deprecated.  Use multi_cla-msg:x_global instead.")
  (x_global m))

(cl:ensure-generic-function 'y_global-val :lambda-list '(m))
(cl:defmethod y_global-val ((m <Location>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader multi_cla-msg:y_global-val is deprecated.  Use multi_cla-msg:y_global instead.")
  (y_global m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Location>) ostream)
  "Serializes a message object of type '<Location>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'x_global))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'y_global))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Location>) istream)
  "Deserializes a message object of type '<Location>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'x_global) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'y_global) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Location>)))
  "Returns string type for a message object of type '<Location>"
  "multi_cla/Location")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Location)))
  "Returns string type for a message object of type 'Location"
  "multi_cla/Location")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Location>)))
  "Returns md5sum for a message object of type '<Location>"
  "7786b133d00488af6ca566640ecc3e11")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Location)))
  "Returns md5sum for a message object of type 'Location"
  "7786b133d00488af6ca566640ecc3e11")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Location>)))
  "Returns full string definition for message of type '<Location>"
  (cl:format cl:nil "float32 x_global~%float32 y_global~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Location)))
  "Returns full string definition for message of type 'Location"
  (cl:format cl:nil "float32 x_global~%float32 y_global~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Location>))
  (cl:+ 0
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Location>))
  "Converts a ROS message object to a list"
  (cl:list 'Location
    (cl:cons ':x_global (x_global msg))
    (cl:cons ':y_global (y_global msg))
))
