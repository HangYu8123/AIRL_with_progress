// Generated by gencpp from file kortex_driver/GetDeviceType.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_GETDEVICETYPE_H
#define KORTEX_DRIVER_MESSAGE_GETDEVICETYPE_H

#include <ros/service_traits.h>


#include <kortex_driver/GetDeviceTypeRequest.h>
#include <kortex_driver/GetDeviceTypeResponse.h>


namespace kortex_driver
{

struct GetDeviceType
{

typedef GetDeviceTypeRequest Request;
typedef GetDeviceTypeResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct GetDeviceType
} // namespace kortex_driver


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::kortex_driver::GetDeviceType > {
  static const char* value()
  {
    return "2d4eec40c5cb478115bd33d8df8d00b7";
  }

  static const char* value(const ::kortex_driver::GetDeviceType&) { return value(); }
};

template<>
struct DataType< ::kortex_driver::GetDeviceType > {
  static const char* value()
  {
    return "kortex_driver/GetDeviceType";
  }

  static const char* value(const ::kortex_driver::GetDeviceType&) { return value(); }
};


// service_traits::MD5Sum< ::kortex_driver::GetDeviceTypeRequest> should match
// service_traits::MD5Sum< ::kortex_driver::GetDeviceType >
template<>
struct MD5Sum< ::kortex_driver::GetDeviceTypeRequest>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::GetDeviceType >::value();
  }
  static const char* value(const ::kortex_driver::GetDeviceTypeRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::GetDeviceTypeRequest> should match
// service_traits::DataType< ::kortex_driver::GetDeviceType >
template<>
struct DataType< ::kortex_driver::GetDeviceTypeRequest>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::GetDeviceType >::value();
  }
  static const char* value(const ::kortex_driver::GetDeviceTypeRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::kortex_driver::GetDeviceTypeResponse> should match
// service_traits::MD5Sum< ::kortex_driver::GetDeviceType >
template<>
struct MD5Sum< ::kortex_driver::GetDeviceTypeResponse>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::GetDeviceType >::value();
  }
  static const char* value(const ::kortex_driver::GetDeviceTypeResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::GetDeviceTypeResponse> should match
// service_traits::DataType< ::kortex_driver::GetDeviceType >
template<>
struct DataType< ::kortex_driver::GetDeviceTypeResponse>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::GetDeviceType >::value();
  }
  static const char* value(const ::kortex_driver::GetDeviceTypeResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_GETDEVICETYPE_H