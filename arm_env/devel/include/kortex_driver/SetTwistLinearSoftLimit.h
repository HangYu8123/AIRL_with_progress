// Generated by gencpp from file kortex_driver/SetTwistLinearSoftLimit.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_SETTWISTLINEARSOFTLIMIT_H
#define KORTEX_DRIVER_MESSAGE_SETTWISTLINEARSOFTLIMIT_H

#include <ros/service_traits.h>


#include <kortex_driver/SetTwistLinearSoftLimitRequest.h>
#include <kortex_driver/SetTwistLinearSoftLimitResponse.h>


namespace kortex_driver
{

struct SetTwistLinearSoftLimit
{

typedef SetTwistLinearSoftLimitRequest Request;
typedef SetTwistLinearSoftLimitResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct SetTwistLinearSoftLimit
} // namespace kortex_driver


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::kortex_driver::SetTwistLinearSoftLimit > {
  static const char* value()
  {
    return "1641ba0fe5229f88d9cda0ee47099b0f";
  }

  static const char* value(const ::kortex_driver::SetTwistLinearSoftLimit&) { return value(); }
};

template<>
struct DataType< ::kortex_driver::SetTwistLinearSoftLimit > {
  static const char* value()
  {
    return "kortex_driver/SetTwistLinearSoftLimit";
  }

  static const char* value(const ::kortex_driver::SetTwistLinearSoftLimit&) { return value(); }
};


// service_traits::MD5Sum< ::kortex_driver::SetTwistLinearSoftLimitRequest> should match
// service_traits::MD5Sum< ::kortex_driver::SetTwistLinearSoftLimit >
template<>
struct MD5Sum< ::kortex_driver::SetTwistLinearSoftLimitRequest>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::SetTwistLinearSoftLimit >::value();
  }
  static const char* value(const ::kortex_driver::SetTwistLinearSoftLimitRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::SetTwistLinearSoftLimitRequest> should match
// service_traits::DataType< ::kortex_driver::SetTwistLinearSoftLimit >
template<>
struct DataType< ::kortex_driver::SetTwistLinearSoftLimitRequest>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::SetTwistLinearSoftLimit >::value();
  }
  static const char* value(const ::kortex_driver::SetTwistLinearSoftLimitRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::kortex_driver::SetTwistLinearSoftLimitResponse> should match
// service_traits::MD5Sum< ::kortex_driver::SetTwistLinearSoftLimit >
template<>
struct MD5Sum< ::kortex_driver::SetTwistLinearSoftLimitResponse>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::SetTwistLinearSoftLimit >::value();
  }
  static const char* value(const ::kortex_driver::SetTwistLinearSoftLimitResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::SetTwistLinearSoftLimitResponse> should match
// service_traits::DataType< ::kortex_driver::SetTwistLinearSoftLimit >
template<>
struct DataType< ::kortex_driver::SetTwistLinearSoftLimitResponse>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::SetTwistLinearSoftLimit >::value();
  }
  static const char* value(const ::kortex_driver::SetTwistLinearSoftLimitResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_SETTWISTLINEARSOFTLIMIT_H