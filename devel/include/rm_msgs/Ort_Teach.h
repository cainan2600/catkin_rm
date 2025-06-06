// Generated by gencpp from file rm_msgs/Ort_Teach.msg
// DO NOT EDIT!


#ifndef RM_MSGS_MESSAGE_ORT_TEACH_H
#define RM_MSGS_MESSAGE_ORT_TEACH_H


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
struct Ort_Teach_
{
  typedef Ort_Teach_<ContainerAllocator> Type;

  Ort_Teach_()
    : teach_type()
    , direction()
    , v(0)  {
    }
  Ort_Teach_(const ContainerAllocator& _alloc)
    : teach_type(_alloc)
    , direction(_alloc)
    , v(0)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _teach_type_type;
  _teach_type_type teach_type;

   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _direction_type;
  _direction_type direction;

   typedef int16_t _v_type;
  _v_type v;





  typedef boost::shared_ptr< ::rm_msgs::Ort_Teach_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rm_msgs::Ort_Teach_<ContainerAllocator> const> ConstPtr;

}; // struct Ort_Teach_

typedef ::rm_msgs::Ort_Teach_<std::allocator<void> > Ort_Teach;

typedef boost::shared_ptr< ::rm_msgs::Ort_Teach > Ort_TeachPtr;
typedef boost::shared_ptr< ::rm_msgs::Ort_Teach const> Ort_TeachConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rm_msgs::Ort_Teach_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rm_msgs::Ort_Teach_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::rm_msgs::Ort_Teach_<ContainerAllocator1> & lhs, const ::rm_msgs::Ort_Teach_<ContainerAllocator2> & rhs)
{
  return lhs.teach_type == rhs.teach_type &&
    lhs.direction == rhs.direction &&
    lhs.v == rhs.v;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::rm_msgs::Ort_Teach_<ContainerAllocator1> & lhs, const ::rm_msgs::Ort_Teach_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace rm_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::Ort_Teach_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rm_msgs::Ort_Teach_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::Ort_Teach_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rm_msgs::Ort_Teach_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::Ort_Teach_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rm_msgs::Ort_Teach_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rm_msgs::Ort_Teach_<ContainerAllocator> >
{
  static const char* value()
  {
    return "34da3e35edafae2adfbdcd46acdb6bd9";
  }

  static const char* value(const ::rm_msgs::Ort_Teach_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x34da3e35edafae2aULL;
  static const uint64_t static_value2 = 0xdfbdcd46acdb6bd9ULL;
};

template<class ContainerAllocator>
struct DataType< ::rm_msgs::Ort_Teach_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rm_msgs/Ort_Teach";
  }

  static const char* value(const ::rm_msgs::Ort_Teach_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rm_msgs::Ort_Teach_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string teach_type\n"
"string direction\n"
"int16 v\n"
;
  }

  static const char* value(const ::rm_msgs::Ort_Teach_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rm_msgs::Ort_Teach_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.teach_type);
      stream.next(m.direction);
      stream.next(m.v);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Ort_Teach_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rm_msgs::Ort_Teach_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rm_msgs::Ort_Teach_<ContainerAllocator>& v)
  {
    s << indent << "teach_type: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.teach_type);
    s << indent << "direction: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.direction);
    s << indent << "v: ";
    Printer<int16_t>::stream(s, indent + "  ", v.v);
  }
};

} // namespace message_operations
} // namespace ros

#endif // RM_MSGS_MESSAGE_ORT_TEACH_H
