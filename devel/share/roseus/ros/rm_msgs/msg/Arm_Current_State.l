;; Auto-generated. Do not edit!


(when (boundp 'rm_msgs::Arm_Current_State)
  (if (not (find-package "RM_MSGS"))
    (make-package "RM_MSGS"))
  (shadow 'Arm_Current_State (find-package "RM_MSGS")))
(unless (find-package "RM_MSGS::ARM_CURRENT_STATE")
  (make-package "RM_MSGS::ARM_CURRENT_STATE"))

(in-package "ROS")
;;//! \htmlinclude Arm_Current_State.msg.html


(defclass rm_msgs::Arm_Current_State
  :super ros::object
  :slots (_joint _Pose _arm_err _sys_err _dof ))

(defmethod rm_msgs::Arm_Current_State
  (:init
   (&key
    ((:joint __joint) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:Pose __Pose) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:arm_err __arm_err) 0)
    ((:sys_err __sys_err) 0)
    ((:dof __dof) 0)
    )
   (send-super :init)
   (setq _joint __joint)
   (setq _Pose __Pose)
   (setq _arm_err (round __arm_err))
   (setq _sys_err (round __sys_err))
   (setq _dof (round __dof))
   self)
  (:joint
   (&optional __joint)
   (if __joint (setq _joint __joint)) _joint)
  (:Pose
   (&optional __Pose)
   (if __Pose (setq _Pose __Pose)) _Pose)
  (:arm_err
   (&optional __arm_err)
   (if __arm_err (setq _arm_err __arm_err)) _arm_err)
  (:sys_err
   (&optional __sys_err)
   (if __sys_err (setq _sys_err __sys_err)) _sys_err)
  (:dof
   (&optional __dof)
   (if __dof (setq _dof __dof)) _dof)
  (:serialization-length
   ()
   (+
    ;; float32[] _joint
    (* 4    (length _joint)) 4
    ;; float32[6] _Pose
    (* 4    6)
    ;; uint16 _arm_err
    2
    ;; uint16 _sys_err
    2
    ;; uint8 _dof
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _joint
     (write-long (length _joint) s)
     (dotimes (i (length _joint))
       (sys::poke (elt _joint i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[6] _Pose
     (dotimes (i 6)
       (sys::poke (elt _Pose i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; uint16 _arm_err
       (write-word _arm_err s)
     ;; uint16 _sys_err
       (write-word _sys_err s)
     ;; uint8 _dof
       (write-byte _dof s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _joint
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _joint (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _joint i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[6] _Pose
   (dotimes (i (length _Pose))
     (setf (elt _Pose i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; uint16 _arm_err
     (setq _arm_err (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _sys_err
     (setq _sys_err (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint8 _dof
     (setq _dof (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(setf (get rm_msgs::Arm_Current_State :md5sum-) "0b0169d007a81fcb8c5cacea4e2efbc9")
(setf (get rm_msgs::Arm_Current_State :datatype-) "rm_msgs/Arm_Current_State")
(setf (get rm_msgs::Arm_Current_State :definition-)
      "float32[] joint
float32[6] Pose
uint16 arm_err
uint16 sys_err
uint8 dof


")



(provide :rm_msgs/Arm_Current_State "0b0169d007a81fcb8c5cacea4e2efbc9")


