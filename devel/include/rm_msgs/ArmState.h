// Generated by gencpp from file rm_msgs/ArmState.msg
// DO NOT EDIT!


#ifndef RM_MSGS_MESSAGE_ARMSTATE_H
#define RM_MSGS_MESSAGE_ARMSTATE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Pose.h>

namespace rm_msgs
{
template <class ContainerAllocator>
struct ArmState_
{
  typedef ArmState_<ContainerAllocator> Type;

  ArmState_()
    : joint()
    , Pose()
    , arm_err(0)
    , sys_err(0)
    , dof(0)  {
    }
  ArmState_(const ContainerAllocator& _alloc)
    : joint(_alloc)
    , Pose(_alloc)
    , arm_err(0)
    , sys_err(0)
    , dof(0)  {
  (void)_alloc;
    }



   typedef std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> _joint_type;
  _joint_type joint;

   typedef  ::geometry_msgs::Pose_<ContainerAllocator>  _Pose_type;
  _Pose_type Pose;

   typedef uint16_t _arm_err_type;
  _arm_err_type arm_err;

   typedef uint16_t _sys_err_type;
  _sys_err_type sys_err;

   typedef uint8_t _dof_type;
  _dof_type dof;





  typedef boost::shared_ptr< ::rm_msgs::ArmState_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rm_msgs::ArmState_<ContainerAllocator> const> ConstPtr;

}; // struct ArmState_

typedef ::rm_msgs::ArmState_<std::allocator<void> > ArmState;

typedef boost::shared_ptr< ::rm_msgs::ArmState > ArmStatePtr;
typedef boost::shared_ptr< ::rm_msgs::ArmState const> ArmStateConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rm_msgs::ArmState_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rm_msgs::ArmState_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::rm_msgs::ArmState_<ContainerAllocator1> & lhs, const ::rm_msgs::ArmState_<ContainerAllocator2> & rhs)
{
  return lhs.joint == rhs.joint &&
    lhs.Pose == rhs.Pose &&
    lhs.arm_err == rhs.arm_err &&
    lhs.sys_err == rhs.sys_err &&
    lhs.dof == rhs.dof;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::rm_msgs::ArmState_<ContainerAllocator1> & lhs, const ::rm_msgs::ArmState_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace rm_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::ArmState_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::ArmState_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::ArmState_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::ArmState_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::ArmState_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::ArmState_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rm_msgs::ArmState_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ba05c85c65cc2ebac457dca171b96eba";
  }

  static const char* value(const ::rm_msgs::ArmState_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xba05c85c65cc2ebaULL;
  static const uint64_t static_value2 = 0xc457dca171b96ebaULL;
};

template<class ContainerAllocator>
struct DataType< ::rm_msgs::ArmState_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rm_msgs/ArmState";
  }

  static const char* value(const ::rm_msgs::ArmState_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rm_msgs::ArmState_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32[] joint\n"
"geometry_msgs/Pose Pose\n"
"uint16 arm_err\n"
"uint16 sys_err\n"
"uint8  dof\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Pose\n"
"# A representation of pose in free space, composed of position and orientation. \n"
"Point position\n"
"Quaternion orientation\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"# This represents an orientation in free space in quaternion form.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
;
  }

  static const char* value(const ::rm_msgs::ArmState_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rm_msgs::ArmState_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.joint);
      stream.next(m.Pose);
      stream.next(m.arm_err);
      stream.next(m.sys_err);
      stream.next(m.dof);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ArmState_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rm_msgs::ArmState_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rm_msgs::ArmState_<ContainerAllocator>& v)
  {
    s << indent << "joint[]" << std::endl;
    for (size_t i = 0; i < v.joint.size(); ++i)
    {
      s << indent << "  joint[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.joint[i]);
    }
    s << indent << "Pose: ";
    s << std::endl;
    Printer< ::geometry_msgs::Pose_<ContainerAllocator> >::stream(s, indent + "  ", v.Pose);
    s << indent << "arm_err: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.arm_err);
    s << indent << "sys_err: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.sys_err);
    s << indent << "dof: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.dof);
  }
};

} // namespace message_operations
} // namespace ros

#endif // RM_MSGS_MESSAGE_ARMSTATE_H
