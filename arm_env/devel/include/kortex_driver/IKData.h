// Generated by gencpp from file kortex_driver/IKData.msg
// DO NOT EDIT!


#ifndef KORTEX_DRIVER_MESSAGE_IKDATA_H
#define KORTEX_DRIVER_MESSAGE_IKDATA_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <kortex_driver/Pose.h>
#include <kortex_driver/JointAngles.h>

namespace kortex_driver
{
template <class ContainerAllocator>
struct IKData_
{
  typedef IKData_<ContainerAllocator> Type;

  IKData_()
    : cartesian_pose()
    , guess()  {
    }
  IKData_(const ContainerAllocator& _alloc)
    : cartesian_pose(_alloc)
    , guess(_alloc)  {
  (void)_alloc;
    }



   typedef  ::kortex_driver::Pose_<ContainerAllocator>  _cartesian_pose_type;
  _cartesian_pose_type cartesian_pose;

   typedef  ::kortex_driver::JointAngles_<ContainerAllocator>  _guess_type;
  _guess_type guess;





  typedef boost::shared_ptr< ::kortex_driver::IKData_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::kortex_driver::IKData_<ContainerAllocator> const> ConstPtr;

}; // struct IKData_

typedef ::kortex_driver::IKData_<std::allocator<void> > IKData;

typedef boost::shared_ptr< ::kortex_driver::IKData > IKDataPtr;
typedef boost::shared_ptr< ::kortex_driver::IKData const> IKDataConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::kortex_driver::IKData_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::kortex_driver::IKData_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::kortex_driver::IKData_<ContainerAllocator1> & lhs, const ::kortex_driver::IKData_<ContainerAllocator2> & rhs)
{
  return lhs.cartesian_pose == rhs.cartesian_pose &&
    lhs.guess == rhs.guess;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::kortex_driver::IKData_<ContainerAllocator1> & lhs, const ::kortex_driver::IKData_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace kortex_driver

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::IKData_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::kortex_driver::IKData_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::IKData_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::kortex_driver::IKData_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::IKData_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::kortex_driver::IKData_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::kortex_driver::IKData_<ContainerAllocator> >
{
  static const char* value()
  {
    return "29f05c9210572828af7df145fee29d3b";
  }

  static const char* value(const ::kortex_driver::IKData_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x29f05c9210572828ULL;
  static const uint64_t static_value2 = 0xaf7df145fee29d3bULL;
};

template<class ContainerAllocator>
struct DataType< ::kortex_driver::IKData_<ContainerAllocator> >
{
  static const char* value()
  {
    return "kortex_driver/IKData";
  }

  static const char* value(const ::kortex_driver::IKData_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::kortex_driver::IKData_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n"
"Pose cartesian_pose\n"
"JointAngles guess\n"
"================================================================================\n"
"MSG: kortex_driver/Pose\n"
"\n"
"float32 x\n"
"float32 y\n"
"float32 z\n"
"float32 theta_x\n"
"float32 theta_y\n"
"float32 theta_z\n"
"================================================================================\n"
"MSG: kortex_driver/JointAngles\n"
"\n"
"JointAngle[] joint_angles\n"
"================================================================================\n"
"MSG: kortex_driver/JointAngle\n"
"\n"
"uint32 joint_identifier\n"
"float32 value\n"
;
  }

  static const char* value(const ::kortex_driver::IKData_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::kortex_driver::IKData_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.cartesian_pose);
      stream.next(m.guess);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct IKData_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::kortex_driver::IKData_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::kortex_driver::IKData_<ContainerAllocator>& v)
  {
    s << indent << "cartesian_pose: ";
    s << std::endl;
    Printer< ::kortex_driver::Pose_<ContainerAllocator> >::stream(s, indent + "  ", v.cartesian_pose);
    s << indent << "guess: ";
    s << std::endl;
    Printer< ::kortex_driver::JointAngles_<ContainerAllocator> >::stream(s, indent + "  ", v.guess);
  }
};

} // namespace message_operations
} // namespace ros

#endif // KORTEX_DRIVER_MESSAGE_IKDATA_H
