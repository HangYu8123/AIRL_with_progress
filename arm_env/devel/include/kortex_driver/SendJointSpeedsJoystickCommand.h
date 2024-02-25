// Generated by gencpp from file kortex_driver/SendJointSpeedsJoystickCommand.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_SENDJOINTSPEEDSJOYSTICKCOMMAND_H
#define KORTEX_DRIVER_MESSAGE_SENDJOINTSPEEDSJOYSTICKCOMMAND_H

#include <ros/service_traits.h>


#include <kortex_driver/SendJointSpeedsJoystickCommandRequest.h>
#include <kortex_driver/SendJointSpeedsJoystickCommandResponse.h>


namespace kortex_driver
{

struct SendJointSpeedsJoystickCommand
{

typedef SendJointSpeedsJoystickCommandRequest Request;
typedef SendJointSpeedsJoystickCommandResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct SendJointSpeedsJoystickCommand
} // namespace kortex_driver


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::kortex_driver::SendJointSpeedsJoystickCommand > {
  static const char* value()
  {
    return "35bff15135e19b4099e6a92d5e7d08d5";
  }

  static const char* value(const ::kortex_driver::SendJointSpeedsJoystickCommand&) { return value(); }
};

template<>
struct DataType< ::kortex_driver::SendJointSpeedsJoystickCommand > {
  static const char* value()
  {
    return "kortex_driver/SendJointSpeedsJoystickCommand";
  }

  static const char* value(const ::kortex_driver::SendJointSpeedsJoystickCommand&) { return value(); }
};


// service_traits::MD5Sum< ::kortex_driver::SendJointSpeedsJoystickCommandRequest> should match
// service_traits::MD5Sum< ::kortex_driver::SendJointSpeedsJoystickCommand >
template<>
struct MD5Sum< ::kortex_driver::SendJointSpeedsJoystickCommandRequest>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::SendJointSpeedsJoystickCommand >::value();
  }
  static const char* value(const ::kortex_driver::SendJointSpeedsJoystickCommandRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::SendJointSpeedsJoystickCommandRequest> should match
// service_traits::DataType< ::kortex_driver::SendJointSpeedsJoystickCommand >
template<>
struct DataType< ::kortex_driver::SendJointSpeedsJoystickCommandRequest>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::SendJointSpeedsJoystickCommand >::value();
  }
  static const char* value(const ::kortex_driver::SendJointSpeedsJoystickCommandRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::kortex_driver::SendJointSpeedsJoystickCommandResponse> should match
// service_traits::MD5Sum< ::kortex_driver::SendJointSpeedsJoystickCommand >
template<>
struct MD5Sum< ::kortex_driver::SendJointSpeedsJoystickCommandResponse>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::SendJointSpeedsJoystickCommand >::value();
  }
  static const char* value(const ::kortex_driver::SendJointSpeedsJoystickCommandResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::SendJointSpeedsJoystickCommandResponse> should match
// service_traits::DataType< ::kortex_driver::SendJointSpeedsJoystickCommand >
template<>
struct DataType< ::kortex_driver::SendJointSpeedsJoystickCommandResponse>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::SendJointSpeedsJoystickCommand >::value();
  }
  static const char* value(const ::kortex_driver::SendJointSpeedsJoystickCommandResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_SENDJOINTSPEEDSJOYSTICKCOMMAND_H
