// Generated by gencpp from file rm_msgs/Tool_IO_State.msg
// DO NOT EDIT!


#ifndef RM_MSGS_MESSAGE_TOOL_IO_STATE_H
#define RM_MSGS_MESSAGE_TOOL_IO_STATE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace rm_msgs
{
template <class ContainerAllocator>
struct Tool_IO_State_
{
  typedef Tool_IO_State_<ContainerAllocator> Type;

  Tool_IO_State_()
    : Tool_IO_Mode()
    , Tool_IO_State()  {
      Tool_IO_Mode.assign(false);

      Tool_IO_State.assign(false);
  }
  Tool_IO_State_(const ContainerAllocator& _alloc)
    : Tool_IO_Mode()
    , Tool_IO_State()  {
  (void)_alloc;
      Tool_IO_Mode.assign(false);

      Tool_IO_State.assign(false);
  }



   typedef boost::array<uint8_t, 2>  _Tool_IO_Mode_type;
  _Tool_IO_Mode_type Tool_IO_Mode;

   typedef boost::array<uint8_t, 2>  _Tool_IO_State_type;
  _Tool_IO_State_type Tool_IO_State;





  typedef boost::shared_ptr< ::rm_msgs::Tool_IO_State_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rm_msgs::Tool_IO_State_<ContainerAllocator> const> ConstPtr;

}; // struct Tool_IO_State_

typedef ::rm_msgs::Tool_IO_State_<std::allocator<void> > Tool_IO_State;

typedef boost::shared_ptr< ::rm_msgs::Tool_IO_State > Tool_IO_StatePtr;
typedef boost::shared_ptr< ::rm_msgs::Tool_IO_State const> Tool_IO_StateConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rm_msgs::Tool_IO_State_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rm_msgs::Tool_IO_State_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::rm_msgs::Tool_IO_State_<ContainerAllocator1> & lhs, const ::rm_msgs::Tool_IO_State_<ContainerAllocator2> & rhs)
{
  return lhs.Tool_IO_Mode == rhs.Tool_IO_Mode &&
    lhs.Tool_IO_State == rhs.Tool_IO_State;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::rm_msgs::Tool_IO_State_<ContainerAllocator1> & lhs, const ::rm_msgs::Tool_IO_State_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace rm_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::Tool_IO_State_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::Tool_IO_State_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::Tool_IO_State_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::Tool_IO_State_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::Tool_IO_State_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::Tool_IO_State_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rm_msgs::Tool_IO_State_<ContainerAllocator> >
{
  static const char* value()
  {
    return "8dedcedb3bfd854b3826d29065f33f9d";
  }

  static const char* value(const ::rm_msgs::Tool_IO_State_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x8dedcedb3bfd854bULL;
  static const uint64_t static_value2 = 0x3826d29065f33f9dULL;
};

template<class ContainerAllocator>
struct DataType< ::rm_msgs::Tool_IO_State_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rm_msgs/Tool_IO_State";
  }

  static const char* value(const ::rm_msgs::Tool_IO_State_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rm_msgs::Tool_IO_State_<ContainerAllocator> >
{
  static const char* value()
  {
    return "bool[2] Tool_IO_Mode          #数字I/O输入/输出状态  0-输入模式，1-输出模式\n"
"bool[2] Tool_IO_State         #数字I/O电平状态      0-低，1-高\n"
;
  }

  static const char* value(const ::rm_msgs::Tool_IO_State_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rm_msgs::Tool_IO_State_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.Tool_IO_Mode);
      stream.next(m.Tool_IO_State);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Tool_IO_State_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rm_msgs::Tool_IO_State_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rm_msgs::Tool_IO_State_<ContainerAllocator>& v)
  {
    s << indent << "Tool_IO_Mode[]" << std::endl;
    for (size_t i = 0; i < v.Tool_IO_Mode.size(); ++i)
    {
      s << indent << "  Tool_IO_Mode[" << i << "]: ";
      Printer<uint8_t>::stream(s, indent + "  ", v.Tool_IO_Mode[i]);
    }
    s << indent << "Tool_IO_State[]" << std::endl;
    for (size_t i = 0; i < v.Tool_IO_State.size(); ++i)
    {
      s << indent << "  Tool_IO_State[" << i << "]: ";
      Printer<uint8_t>::stream(s, indent + "  ", v.Tool_IO_State[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // RM_MSGS_MESSAGE_TOOL_IO_STATE_H