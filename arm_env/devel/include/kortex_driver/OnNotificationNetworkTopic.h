// Generated by gencpp from file kortex_driver/OnNotificationNetworkTopic.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_ONNOTIFICATIONNETWORKTOPIC_H
#define KORTEX_DRIVER_MESSAGE_ONNOTIFICATIONNETWORKTOPIC_H

#include <ros/service_traits.h>


#include <kortex_driver/OnNotificationNetworkTopicRequest.h>
#include <kortex_driver/OnNotificationNetworkTopicResponse.h>


namespace kortex_driver
{

struct OnNotificationNetworkTopic
{

typedef OnNotificationNetworkTopicRequest Request;
typedef OnNotificationNetworkTopicResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct OnNotificationNetworkTopic
} // namespace kortex_driver


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::kortex_driver::OnNotificationNetworkTopic > {
  static const char* value()
  {
    return "6fefdd07c6cb63a94f7b48e7e07e815b";
  }

  static const char* value(const ::kortex_driver::OnNotificationNetworkTopic&) { return value(); }
};

template<>
struct DataType< ::kortex_driver::OnNotificationNetworkTopic > {
  static const char* value()
  {
    return "kortex_driver/OnNotificationNetworkTopic";
  }

  static const char* value(const ::kortex_driver::OnNotificationNetworkTopic&) { return value(); }
};


// service_traits::MD5Sum< ::kortex_driver::OnNotificationNetworkTopicRequest> should match
// service_traits::MD5Sum< ::kortex_driver::OnNotificationNetworkTopic >
template<>
struct MD5Sum< ::kortex_driver::OnNotificationNetworkTopicRequest>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::OnNotificationNetworkTopic >::value();
  }
  static const char* value(const ::kortex_driver::OnNotificationNetworkTopicRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::OnNotificationNetworkTopicRequest> should match
// service_traits::DataType< ::kortex_driver::OnNotificationNetworkTopic >
template<>
struct DataType< ::kortex_driver::OnNotificationNetworkTopicRequest>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::OnNotificationNetworkTopic >::value();
  }
  static const char* value(const ::kortex_driver::OnNotificationNetworkTopicRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::kortex_driver::OnNotificationNetworkTopicResponse> should match
// service_traits::MD5Sum< ::kortex_driver::OnNotificationNetworkTopic >
template<>
struct MD5Sum< ::kortex_driver::OnNotificationNetworkTopicResponse>
{
  static const char* value()
  {
    return MD5Sum< ::kortex_driver::OnNotificationNetworkTopic >::value();
  }
  static const char* value(const ::kortex_driver::OnNotificationNetworkTopicResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::kortex_driver::OnNotificationNetworkTopicResponse> should match
// service_traits::DataType< ::kortex_driver::OnNotificationNetworkTopic >
template<>
struct DataType< ::kortex_driver::OnNotificationNetworkTopicResponse>
{
  static const char* value()
  {
    return DataType< ::kortex_driver::OnNotificationNetworkTopic >::value();
  }
  static const char* value(const ::kortex_driver::OnNotificationNetworkTopicResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_ONNOTIFICATIONNETWORKTOPIC_H