// Generated by gencpp from file kortex_driver/UpdateUserProfileRequest.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_UPDATEUSERPROFILEREQUEST_H
#define KORTEX_DRIVER_MESSAGE_UPDATEUSERPROFILEREQUEST_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <kortex_driver/UserProfile.h>

namespace kortex_driver
{
template <class ContainerAllocator>
struct UpdateUserProfileRequest_
{
  typedef UpdateUserProfileRequest_<ContainerAllocator> Type;

  UpdateUserProfileRequest_()
    : input()  {
    }
  UpdateUserProfileRequest_(const ContainerAllocator& _alloc)
    : input(_alloc)  {
  (void)_alloc;
    }



   typedef  ::kortex_driver::UserProfile_<ContainerAllocator>  _input_type;
  _input_type input;





  typedef boost::shared_ptr< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> const> ConstPtr;

}; // struct UpdateUserProfileRequest_

typedef ::kortex_driver::UpdateUserProfileRequest_<std::allocator<void> > UpdateUserProfileRequest;

typedef boost::shared_ptr< ::kortex_driver::UpdateUserProfileRequest > UpdateUserProfileRequestPtr;
typedef boost::shared_ptr< ::kortex_driver::UpdateUserProfileRequest const> UpdateUserProfileRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator1> & lhs, const ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator2> & rhs)
{
  return lhs.input == rhs.input;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator1> & lhs, const ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace kortex_driver

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "759cfa6ab6da4b488c7a1ac251741de6";
  }

  static const char* value(const ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x759cfa6ab6da4b48ULL;
  static const uint64_t static_value2 = 0x8c7a1ac251741de6ULL;
};

template<class ContainerAllocator>
struct DataType< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "kortex_driver/UpdateUserProfileRequest";
  }

  static const char* value(const ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "UserProfile input\n"
"\n"
"================================================================================\n"
"MSG: kortex_driver/UserProfile\n"
"\n"
"UserProfileHandle handle\n"
"string username\n"
"string firstname\n"
"string lastname\n"
"string application_data\n"
"================================================================================\n"
"MSG: kortex_driver/UserProfileHandle\n"
"\n"
"uint32 identifier\n"
"uint32 permission\n"
;
  }

  static const char* value(const ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.input);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct UpdateUserProfileRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::kortex_driver::UpdateUserProfileRequest_<ContainerAllocator>& v)
  {
    s << indent << "input: ";
    s << std::endl;
    Printer< ::kortex_driver::UserProfile_<ContainerAllocator> >::stream(s, indent + "  ", v.input);
  }
};

} // namespace message_operations
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_UPDATEUSERPROFILEREQUEST_H