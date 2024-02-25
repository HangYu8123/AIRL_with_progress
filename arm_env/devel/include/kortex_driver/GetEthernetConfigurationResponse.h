// Generated by gencpp from file kortex_driver/GetEthernetConfigurationResponse.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_GETETHERNETCONFIGURATIONRESPONSE_H
#define KORTEX_DRIVER_MESSAGE_GETETHERNETCONFIGURATIONRESPONSE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <kortex_driver/EthernetConfiguration.h>

namespace kortex_driver
{
template <class ContainerAllocator>
struct GetEthernetConfigurationResponse_
{
  typedef GetEthernetConfigurationResponse_<ContainerAllocator> Type;

  GetEthernetConfigurationResponse_()
    : output()  {
    }
  GetEthernetConfigurationResponse_(const ContainerAllocator& _alloc)
    : output(_alloc)  {
  (void)_alloc;
    }



   typedef  ::kortex_driver::EthernetConfiguration_<ContainerAllocator>  _output_type;
  _output_type output;





  typedef boost::shared_ptr< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> const> ConstPtr;

}; // struct GetEthernetConfigurationResponse_

typedef ::kortex_driver::GetEthernetConfigurationResponse_<std::allocator<void> > GetEthernetConfigurationResponse;

typedef boost::shared_ptr< ::kortex_driver::GetEthernetConfigurationResponse > GetEthernetConfigurationResponsePtr;
typedef boost::shared_ptr< ::kortex_driver::GetEthernetConfigurationResponse const> GetEthernetConfigurationResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator1> & lhs, const ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator2> & rhs)
{
  return lhs.output == rhs.output;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator1> & lhs, const ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace kortex_driver

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "62205af0ba461c567072364c0b0527fe";
  }

  static const char* value(const ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x62205af0ba461c56ULL;
  static const uint64_t static_value2 = 0x7072364c0b0527feULL;
};

template<class ContainerAllocator>
struct DataType< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "kortex_driver/GetEthernetConfigurationResponse";
  }

  static const char* value(const ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "EthernetConfiguration output\n"
"\n"
"================================================================================\n"
"MSG: kortex_driver/EthernetConfiguration\n"
"\n"
"uint32 device\n"
"bool enabled\n"
"uint32 speed\n"
"uint32 duplex\n"
;
  }

  static const char* value(const ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.output);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct GetEthernetConfigurationResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::kortex_driver::GetEthernetConfigurationResponse_<ContainerAllocator>& v)
  {
    s << indent << "output: ";
    s << std::endl;
    Printer< ::kortex_driver::EthernetConfiguration_<ContainerAllocator> >::stream(s, indent + "  ", v.output);
  }
};

} // namespace message_operations
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_GETETHERNETCONFIGURATIONRESPONSE_H
