// Generated by gencpp from file kortex_driver/UpdateSequence.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_UPDATESEQUENCE_H
#define KORTEX_DRIVER_MESSAGE_UPDATESEQUENCE_H

#include <ros/service_traits.h>


#include <kortex_driver/UpdateSequenceRequest.h>
#include <kortex_driver/UpdateSequenceResponse.h>


namespace kortex_driver
{

struct UpdateSequence
{

typedef UpdateSequenceRequest Request;
typedef UpdateSequenceResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct UpdateSequence
} // namespace kortex_driver


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::kortex_driver::UpdateSequence > {
  static const char* value()
  {
    return "42cb8fcf59e13a93c2ae4b3f1a8f8519";
  }

  static const char* value(const ::kortex_driver::UpdateSequence&) { return value(); }
};

template<>
struct DataType< ::kortex_driver::UpdateSequence > {
  static const char* value()
  {
    return "kortex_driver/UpdateSequence";
  }

  static const char* value(const ::kortex_driver::UpdateSequence&) { return value(); }
};


// service_traits::MD5Sum< ::kortex_driver::UpdateSequenceRequest> should match
// service_traits::MD5Sum< ::kortex_driver::UpdateSequence >
template<>
struct MD5Sum< ::kortex_driver::UpdateSequenceRequest>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::UpdateSequence >::value();
  }
  static const char* value(const ::kortex_driver::UpdateSequenceRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::UpdateSequenceRequest> should match
// service_traits::DataType< ::kortex_driver::UpdateSequence >
template<>
struct DataType< ::kortex_driver::UpdateSequenceRequest>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::UpdateSequence >::value();
  }
  static const char* value(const ::kortex_driver::UpdateSequenceRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::kortex_driver::UpdateSequenceResponse> should match
// service_traits::MD5Sum< ::kortex_driver::UpdateSequence >
template<>
struct MD5Sum< ::kortex_driver::UpdateSequenceResponse>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::UpdateSequence >::value();
  }
  static const char* value(const ::kortex_driver::UpdateSequenceResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::UpdateSequenceResponse> should match
// service_traits::DataType< ::kortex_driver::UpdateSequence >
template<>
struct DataType< ::kortex_driver::UpdateSequenceResponse>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::UpdateSequence >::value();
  }
  static const char* value(const ::kortex_driver::UpdateSequenceResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_UPDATESEQUENCE_H
