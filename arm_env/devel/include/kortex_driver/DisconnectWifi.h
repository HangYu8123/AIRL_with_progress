// Generated by gencpp from file kortex_driver/DisconnectWifi.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_DISCONNECTWIFI_H
#define KORTEX_DRIVER_MESSAGE_DISCONNECTWIFI_H

#include <ros/service_traits.h>


#include <kortex_driver/DisconnectWifiRequest.h>
#include <kortex_driver/DisconnectWifiResponse.h>


namespace kortex_driver
{

struct DisconnectWifi
{

typedef DisconnectWifiRequest Request;
typedef DisconnectWifiResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct DisconnectWifi
} // namespace kortex_driver


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::kortex_driver::DisconnectWifi > {
  static const char* value()
  {
    return "f335b819dc59099fe3124b36f140ad07";
  }

  static const char* value(const ::kortex_driver::DisconnectWifi&) { return value(); }
};

template<>
struct DataType< ::kortex_driver::DisconnectWifi > {
  static const char* value()
  {
    return "kortex_driver/DisconnectWifi";
  }

  static const char* value(const ::kortex_driver::DisconnectWifi&) { return value(); }
};


// service_traits::MD5Sum< ::kortex_driver::DisconnectWifiRequest> should match
// service_traits::MD5Sum< ::kortex_driver::DisconnectWifi >
template<>
struct MD5Sum< ::kortex_driver::DisconnectWifiRequest>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::DisconnectWifi >::value();
  }
  static const char* value(const ::kortex_driver::DisconnectWifiRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::DisconnectWifiRequest> should match
// service_traits::DataType< ::kortex_driver::DisconnectWifi >
template<>
struct DataType< ::kortex_driver::DisconnectWifiRequest>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::DisconnectWifi >::value();
  }
  static const char* value(const ::kortex_driver::DisconnectWifiRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::kortex_driver::DisconnectWifiResponse> should match
// service_traits::MD5Sum< ::kortex_driver::DisconnectWifi >
template<>
struct MD5Sum< ::kortex_driver::DisconnectWifiResponse>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::DisconnectWifi >::value();
  }
  static const char* value(const ::kortex_driver::DisconnectWifiResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::DisconnectWifiResponse> should match
// service_traits::DataType< ::kortex_driver::DisconnectWifi >
template<>
struct DataType< ::kortex_driver::DisconnectWifiResponse>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::DisconnectWifi >::value();
  }
  static const char* value(const ::kortex_driver::DisconnectWifiResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_DISCONNECTWIFI_H
