;; Auto-generated. Do not edit!


(when (boundp 'multi_cla::Location)
  (if (not (find-package "MULTI_CLA"))
    (make-package "MULTI_CLA"))
  (shadow 'Location (find-package "MULTI_CLA")))
(unless (find-package "MULTI_CLA::LOCATION")
  (make-package "MULTI_CLA::LOCATION"))

(in-package "ROS")
;;//! \htmlinclude Location.msg.html


(defclass multi_cla::Location
  :super ros::object
  :slots (_x_global _y_global ))

(defmethod multi_cla::Location
  (:init
   (&key
    ((:x_global __x_global) 0.0)
    ((:y_global __y_global) 0.0)
    )
   (send-super :init)
   (setq _x_global (float __x_global))
   (setq _y_global (float __y_global))
   self)
  (:x_global
   (&optional __x_global)
   (if __x_global (setq _x_global __x_global)) _x_global)
  (:y_global
   (&optional __y_global)
   (if __y_global (setq _y_global __y_global)) _y_global)
  (:serialization-length
   ()
   (+
    ;; float32 _x_global
    4
    ;; float32 _y_global
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _x_global
       (sys::poke _x_global (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _y_global
       (sys::poke _y_global (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _x_global
     (setq _x_global (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _y_global
     (setq _y_global (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get multi_cla::Location :md5sum-) "7786b133d00488af6ca566640ecc3e11")
(setf (get multi_cla::Location :datatype-) "multi_cla/Location")
(setf (get multi_cla::Location :definition-)
      "float32 x_global
float32 y_global

")



(provide :multi_cla/Location "7786b133d00488af6ca566640ecc3e11")


