;; Auto-generated. Do not edit!


(when (boundp 'rm_msgs::MoveC)
  (if (not (find-package "RM_MSGS"))
    (make-package "RM_MSGS"))
  (shadow 'MoveC (find-package "RM_MSGS")))
(unless (find-package "RM_MSGS::MOVEC")
  (make-package "RM_MSGS::MOVEC"))

(in-package "ROS")
;;//! \htmlinclude MoveC.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass rm_msgs::MoveC
  :super ros::object
  :slots (_Mid_Pose _End_Pose _speed _loop _trajectory_connect ))

(defmethod rm_msgs::MoveC
  (:init
   (&key
    ((:Mid_Pose __Mid_Pose) (instance geometry_msgs::Pose :init))
    ((:End_Pose __End_Pose) (instance geometry_msgs::Pose :init))
    ((:speed __speed) 0.0)
    ((:loop __loop) 0)
    ((:trajectory_connect __trajectory_connect) 0)
    )
   (send-super :init)
   (setq _Mid_Pose __Mid_Pose)
   (setq _End_Pose __End_Pose)
   (setq _speed (float __speed))
   (setq _loop (round __loop))
   (setq _trajectory_connect (round __trajectory_connect))
   self)
  (:Mid_Pose
   (&rest __Mid_Pose)
   (if (keywordp (car __Mid_Pose))
       (send* _Mid_Pose __Mid_Pose)
     (progn
       (if __Mid_Pose (setq _Mid_Pose (car __Mid_Pose)))
       _Mid_Pose)))
  (:End_Pose
   (&rest __End_Pose)
   (if (keywordp (car __End_Pose))
       (send* _End_Pose __End_Pose)
     (progn
       (if __End_Pose (setq _End_Pose (car __End_Pose)))
       _End_Pose)))
  (:speed
   (&optional __speed)
   (if __speed (setq _speed __speed)) _speed)
  (:loop
   (&optional __loop)
   (if __loop (setq _loop __loop)) _loop)
  (:trajectory_connect
   (&optional __trajectory_connect)
   (if __trajectory_connect (setq _trajectory_connect __trajectory_connect)) _trajectory_connect)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Pose _Mid_Pose
    (send _Mid_Pose :serialization-length)
    ;; geometry_msgs/Pose _End_Pose
    (send _End_Pose :serialization-length)
    ;; float32 _speed
    4
    ;; uint16 _loop
    2
    ;; uint8 _trajectory_connect
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Pose _Mid_Pose
       (send _Mid_Pose :serialize s)
     ;; geometry_msgs/Pose _End_Pose
       (send _End_Pose :serialize s)
     ;; float32 _speed
       (sys::poke _speed (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint16 _loop
       (write-word _loop s)
     ;; uint8 _trajectory_connect
       (write-byte _trajectory_connect s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Pose _Mid_Pose
     (send _Mid_Pose :deserialize buf ptr-) (incf ptr- (send _Mid_Pose :serialization-length))
   ;; geometry_msgs/Pose _End_Pose
     (send _End_Pose :deserialize buf ptr-) (incf ptr- (send _End_Pose :serialization-length))
   ;; float32 _speed
     (setq _speed (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint16 _loop
     (setq _loop (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint8 _trajectory_connect
     (setq _trajectory_connect (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(setf (get rm_msgs::MoveC :md5sum-) "4a51b555e13bfa444df1e7d9863b0ad6")
(setf (get rm_msgs::MoveC :datatype-) "rm_msgs/MoveC")
(setf (get rm_msgs::MoveC :definition-)
      "geometry_msgs/Pose Mid_Pose
geometry_msgs/Pose End_Pose
float32 speed
uint16 loop
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

")



(provide :rm_msgs/MoveC "4a51b555e13bfa444df1e7d9863b0ad6")


