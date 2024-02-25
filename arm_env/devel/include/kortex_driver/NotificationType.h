// Generated by gencpp from file kortex_driver/NotificationType.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_NOTIFICATIONTYPE_H
#define KORTEX_DRIVER_MESSAGE_NOTIFICATIONTYPE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace kortex_driver
{
template <class ContainerAllocator>
struct NotificationType_
{
  typedef NotificationType_<ContainerAllocator> Type;

  NotificationType_()
    {
    }
  NotificationType_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }





// reducing the odds to have name collisions with Windows.h 
#if defined(_WIN32) && defined(NOTIFICATION_TYPE_UNSPECIFIED)
  #undef NOTIFICATION_TYPE_UNSPECIFIED
#endif
#if defined(_WIN32) && defined(NOTIFICATION_TYPE_THRESHOLD)
  #undef NOTIFICATION_TYPE_THRESHOLD
#endif
#if defined(_WIN32) && defined(NOTIFICATION_TYPE_FIX_RATE)
  #undef NOTIFICATION_TYPE_FIX_RATE
#endif
#if defined(_WIN32) && defined(NOTIFICATION_TYPE_EVENT)
  #undef NOTIFICATION_TYPE_EVENT
#endif

  enum {
    NOTIFICATION_TYPE_UNSPECIFIED = 0u,
    NOTIFICATION_TYPE_THRESHOLD = 1u,
    NOTIFICATION_TYPE_FIX_RATE = 2u,
    NOTIFICATION_TYPE_EVENT = 3u,
  };


  typedef boost::shared_ptr< ::kortex_driver::NotificationType_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::kortex_driver::NotificationType_<ContainerAllocator> const> ConstPtr;

}; // struct NotificationType_

typedef ::kortex_driver::NotificationType_<std::allocator<void> > NotificationType;

typedef boost::shared_ptr< ::kortex_driver::NotificationType > NotificationTypePtr;
typedef boost::shared_ptr< ::kortex_driver::NotificationType const> NotificationTypeConstPtr;

// constants requiring out of line definition

   

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::kortex_driver::NotificationType_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::kortex_driver::NotificationType_<ContainerAllocator> >::stream(s, "", v);
return s;
}


} // namespace kortex_driver

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::NotificationType_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::NotificationType_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::NotificationType_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::NotificationType_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::NotificationType_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::NotificationType_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::kortex_driver::NotificationType_<ContainerAllocator> >
{
  static const char* value()
  {
    return "9d8153f0fe98641698596673829b2649";
  }

  static const char* value(const ::kortex_driver::NotificationType_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x9d8153f0fe986416ULL;
  static const uint64_t static_value2 = 0x98596673829b2649ULL;
};

template<class ContainerAllocator>
struct DataType< ::kortex_driver::NotificationType_<ContainerAllocator> >
{
  static const char* value()
  {
    return "kortex_driver/NotificationType";
  }

  static const char* value(const ::kortex_driver::NotificationType_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::kortex_driver::NotificationType_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n"
"uint32 NOTIFICATION_TYPE_UNSPECIFIED = 0\n"
"\n"
"uint32 NOTIFICATION_TYPE_THRESHOLD = 1\n"
"\n"
"uint32 NOTIFICATION_TYPE_FIX_RATE = 2\n"
"\n"
"uint32 NOTIFICATION_TYPE_EVENT = 3\n"
;
  }

  static const char* value(const ::kortex_driver::NotificationType_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::kortex_driver::NotificationType_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct NotificationType_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::kortex_driver::NotificationType_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::kortex_driver::NotificationType_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_NOTIFICATIONTYPE_H
