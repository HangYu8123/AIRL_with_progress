// Generated by gencpp from file kortex_driver/ExecuteActionFromReference.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_EXECUTEACTIONFROMREFERENCE_H
#define KORTEX_DRIVER_MESSAGE_EXECUTEACTIONFROMREFERENCE_H

#include <ros/service_traits.h>


#include <kortex_driver/ExecuteActionFromReferenceRequest.h>
#include <kortex_driver/ExecuteActionFromReferenceResponse.h>


namespace kortex_driver
{

struct ExecuteActionFromReference
{

typedef ExecuteActionFromReferenceRequest Request;
typedef ExecuteActionFromReferenceResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct ExecuteActionFromReference
} // namespace kortex_driver


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::kortex_driver::ExecuteActionFromReference > {
  static const char* value()
  {
    return "39696246fa7132aebfa0097dedbf54c1";
  }

  static const char* value(const ::kortex_driver::ExecuteActionFromReference&) { return value(); }
};

template<>
struct DataType< ::kortex_driver::ExecuteActionFromReference > {
  static const char* value()
  {
    return "kortex_driver/ExecuteActionFromReference";
  }

  static const char* value(const ::kortex_driver::ExecuteActionFromReference&) { return value(); }
};


// service_traits::MD5Sum< ::kortex_driver::ExecuteActionFromReferenceRequest> should match
// service_traits::MD5Sum< ::kortex_driver::ExecuteActionFromReference >
template<>
struct MD5Sum< ::kortex_driver::ExecuteActionFromReferenceRequest>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::ExecuteActionFromReference >::value();
  }
  static const char* value(const ::kortex_driver::ExecuteActionFromReferenceRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::ExecuteActionFromReferenceRequest> should match
// service_traits::DataType< ::kortex_driver::ExecuteActionFromReference >
template<>
struct DataType< ::kortex_driver::ExecuteActionFromReferenceRequest>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::ExecuteActionFromReference >::value();
  }
  static const char* value(const ::kortex_driver::ExecuteActionFromReferenceRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::kortex_driver::ExecuteActionFromReferenceResponse> should match
// service_traits::MD5Sum< ::kortex_driver::ExecuteActionFromReference >
template<>
struct MD5Sum< ::kortex_driver::ExecuteActionFromReferenceResponse>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::ExecuteActionFromReference >::value();
  }
  static const char* value(const ::kortex_driver::ExecuteActionFromReferenceResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::ExecuteActionFromReferenceResponse> should match
// service_traits::DataType< ::kortex_driver::ExecuteActionFromReference >
template<>
struct DataType< ::kortex_driver::ExecuteActionFromReferenceResponse>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::ExecuteActionFromReference >::value();
  }
  static const char* value(const ::kortex_driver::ExecuteActionFromReferenceResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_EXECUTEACTIONFROMREFERENCE_H