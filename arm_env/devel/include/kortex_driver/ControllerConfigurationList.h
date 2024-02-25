// Generated by gencpp from file kortex_driver/ControllerConfigurationList.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_CONTROLLERCONFIGURATIONLIST_H
#define KORTEX_DRIVER_MESSAGE_CONTROLLERCONFIGURATIONLIST_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <kortex_driver/ControllerConfiguration.h>

namespace kortex_driver
{
template <class ContainerAllocator>
struct ControllerConfigurationList_
{
  typedef ControllerConfigurationList_<ContainerAllocator> Type;

  ControllerConfigurationList_()
    : controller_configurations()  {
    }
  ControllerConfigurationList_(const ContainerAllocator& _alloc)
    : controller_configurations(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector< ::kortex_driver::ControllerConfiguration_<ContainerAllocator> , typename std::allocator_traits<ContainerAllocator>::template rebind_alloc< ::kortex_driver::ControllerConfiguration_<ContainerAllocator> >> _controller_configurations_type;
  _controller_configurations_type controller_configurations;





  typedef boost::shared_ptr< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> const> ConstPtr;

}; // struct ControllerConfigurationList_

typedef ::kortex_driver::ControllerConfigurationList_<std::allocator<void> > ControllerConfigurationList;

typedef boost::shared_ptr< ::kortex_driver::ControllerConfigurationList > ControllerConfigurationListPtr;
typedef boost::shared_ptr< ::kortex_driver::ControllerConfigurationList const> ControllerConfigurationListConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::kortex_driver::ControllerConfigurationList_<ContainerAllocator1> & lhs, const ::kortex_driver::ControllerConfigurationList_<ContainerAllocator2> & rhs)
{
  return lhs.controller_configurations == rhs.controller_configurations;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::kortex_driver::ControllerConfigurationList_<ContainerAllocator1> & lhs, const ::kortex_driver::ControllerConfigurationList_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace kortex_driver

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> >
{
  static const char* value()
  {
    return "4e505f81204befaff98df48e637201d5";
  }

  static const char* value(const ::kortex_driver::ControllerConfigurationList_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x4e505f81204befafULL;
  static const uint64_t static_value2 = 0xf98df48e637201d5ULL;
};

template<class ContainerAllocator>
struct DataType< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> >
{
  static const char* value()
  {
    return "kortex_driver/ControllerConfigurationList";
  }

  static const char* value(const ::kortex_driver::ControllerConfigurationList_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n"
"ControllerConfiguration[] controller_configurations\n"
"================================================================================\n"
"MSG: kortex_driver/ControllerConfiguration\n"
"\n"
"ControllerHandle handle\n"
"string name\n"
"MappingHandle active_mapping_handle\n"
"string analog_input_identifier_enum\n"
"string digital_input_identifier_enum\n"
"================================================================================\n"
"MSG: kortex_driver/ControllerHandle\n"
"\n"
"uint32 type\n"
"uint32 controller_identifier\n"
"================================================================================\n"
"MSG: kortex_driver/MappingHandle\n"
"\n"
"uint32 identifier\n"
"uint32 permission\n"
;
  }

  static const char* value(const ::kortex_driver::ControllerConfigurationList_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.controller_configurations);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ControllerConfigurationList_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::kortex_driver::ControllerConfigurationList_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::kortex_driver::ControllerConfigurationList_<ContainerAllocator>& v)
  {
    s << indent << "controller_configurations[]" << std::endl;
    for (size_t i = 0; i < v.controller_configurations.size(); ++i)
    {
      s << indent << "  controller_configurations[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::kortex_driver::ControllerConfiguration_<ContainerAllocator> >::stream(s, indent + "    ", v.controller_configurations[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_CONTROLLERCONFIGURATIONLIST_H
