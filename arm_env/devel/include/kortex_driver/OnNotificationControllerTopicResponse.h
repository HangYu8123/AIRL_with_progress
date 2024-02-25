// Generated by gencpp from file kortex_driver/OnNotificationControllerTopicResponse.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_ONNOTIFICATIONCONTROLLERTOPICRESPONSE_H
#define KORTEX_DRIVER_MESSAGE_ONNOTIFICATIONCONTROLLERTOPICRESPONSE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <kortex_driver/NotificationHandle.h>

namespace kortex_driver
{
template <class ContainerAllocator>
struct OnNotificationControllerTopicResponse_
{
  typedef OnNotificationControllerTopicResponse_<ContainerAllocator> Type;

  OnNotificationControllerTopicResponse_()
    : output()  {
    }
  OnNotificationControllerTopicResponse_(const ContainerAllocator& _alloc)
    : output(_alloc)  {
  (void)_alloc;
    }



   typedef  ::kortex_driver::NotificationHandle_<ContainerAllocator>  _output_type;
  _output_type output;





  typedef boost::shared_ptr< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> const> ConstPtr;

}; // struct OnNotificationControllerTopicResponse_

typedef ::kortex_driver::OnNotificationControllerTopicResponse_<std::allocator<void> > OnNotificationControllerTopicResponse;

typedef boost::shared_ptr< ::kortex_driver::OnNotificationControllerTopicResponse > OnNotificationControllerTopicResponsePtr;
typedef boost::shared_ptr< ::kortex_driver::OnNotificationControllerTopicResponse const> OnNotificationControllerTopicResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator1> & lhs, const ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator2> & rhs)
{
  return lhs.output == rhs.output;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator1> & lhs, const ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace kortex_driver

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "29ff9348c5c8343d487a90668267a29e";
  }

  static const char* value(const ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x29ff9348c5c8343dULL;
  static const uint64_t static_value2 = 0x487a90668267a29eULL;
};

template<class ContainerAllocator>
struct DataType< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "kortex_driver/OnNotificationControllerTopicResponse";
  }

  static const char* value(const ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "NotificationHandle output\n"
"\n"
"================================================================================\n"
"MSG: kortex_driver/NotificationHandle\n"
"\n"
"uint32 identifier\n"
;
  }

  static const char* value(const ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.output);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct OnNotificationControllerTopicResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::kortex_driver::OnNotificationControllerTopicResponse_<ContainerAllocator>& v)
  {
    s << indent << "output: ";
    s << std::endl;
    Printer< ::kortex_driver::NotificationHandle_<ContainerAllocator> >::stream(s, indent + "  ", v.output);
  }
};

} // namespace message_operations
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_ONNOTIFICATIONCONTROLLERTOPICRESPONSE_H
