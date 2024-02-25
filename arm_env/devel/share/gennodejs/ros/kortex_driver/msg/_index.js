
"use strict";

let ApiOptions = require('./ApiOptions.js');
let KortexError = require('./KortexError.js');
let ErrorCodes = require('./ErrorCodes.js');
let SubErrorCodes = require('./SubErrorCodes.js');
let CoggingFeedforwardModeInformation = require('./CoggingFeedforwardModeInformation.js');
let ActuatorConfig_ServiceVersion = require('./ActuatorConfig_ServiceVersion.js');
let EncoderDerivativeParameters = require('./EncoderDerivativeParameters.js');
let CoggingFeedforwardMode = require('./CoggingFeedforwardMode.js');
let FrequencyResponse = require('./FrequencyResponse.js');
let CustomDataSelection = require('./CustomDataSelection.js');
let ControlLoopSelection = require('./ControlLoopSelection.js');
let ControlLoop = require('./ControlLoop.js');
let VectorDriveParameters = require('./VectorDriveParameters.js');
let CommandMode = require('./CommandMode.js');
let StepResponse = require('./StepResponse.js');
let AxisPosition = require('./AxisPosition.js');
let Servoing = require('./Servoing.js');
let CommandModeInformation = require('./CommandModeInformation.js');
let SafetyIdentifierBankA = require('./SafetyIdentifierBankA.js');
let RampResponse = require('./RampResponse.js');
let PositionCommand = require('./PositionCommand.js');
let ControlLoopParameters = require('./ControlLoopParameters.js');
let TorqueCalibration = require('./TorqueCalibration.js');
let AxisOffsets = require('./AxisOffsets.js');
let CustomDataIndex = require('./CustomDataIndex.js');
let ActuatorConfig_SafetyLimitType = require('./ActuatorConfig_SafetyLimitType.js');
let ActuatorConfig_ControlModeInformation = require('./ActuatorConfig_ControlModeInformation.js');
let LoopSelection = require('./LoopSelection.js');
let TorqueOffset = require('./TorqueOffset.js');
let ActuatorConfig_ControlMode = require('./ActuatorConfig_ControlMode.js');
let ActuatorCyclic_Command = require('./ActuatorCyclic_Command.js');
let CommandFlags = require('./CommandFlags.js');
let ActuatorCyclic_Feedback = require('./ActuatorCyclic_Feedback.js');
let ActuatorCyclic_CustomData = require('./ActuatorCyclic_CustomData.js');
let StatusFlags = require('./StatusFlags.js');
let ActuatorCyclic_MessageId = require('./ActuatorCyclic_MessageId.js');
let ActuatorCyclic_ServiceVersion = require('./ActuatorCyclic_ServiceVersion.js');
let RobotEvent = require('./RobotEvent.js');
let MapEvent = require('./MapEvent.js');
let CartesianTrajectoryConstraint_type = require('./CartesianTrajectoryConstraint_type.js');
let Base_ServiceVersion = require('./Base_ServiceVersion.js');
let Ssid = require('./Ssid.js');
let CartesianLimitation = require('./CartesianLimitation.js');
let BridgeIdentifier = require('./BridgeIdentifier.js');
let Query = require('./Query.js');
let ServoingModeInformation = require('./ServoingModeInformation.js');
let ProtectionZoneList = require('./ProtectionZoneList.js');
let Pose = require('./Pose.js');
let AppendActionInformation = require('./AppendActionInformation.js');
let JointNavigationDirection = require('./JointNavigationDirection.js');
let MapGroupList = require('./MapGroupList.js');
let KinematicTrajectoryConstraints = require('./KinematicTrajectoryConstraints.js');
let Map = require('./Map.js');
let JointTrajectoryConstraint = require('./JointTrajectoryConstraint.js');
let WaypointList = require('./WaypointList.js');
let RobotEventNotificationList = require('./RobotEventNotificationList.js');
let ShapeType = require('./ShapeType.js');
let ConfigurationChangeNotificationList = require('./ConfigurationChangeNotificationList.js');
let ControllerState = require('./ControllerState.js');
let TrajectoryInfoType = require('./TrajectoryInfoType.js');
let SequenceTasks = require('./SequenceTasks.js');
let CartesianWaypoint = require('./CartesianWaypoint.js');
let GripperRequest = require('./GripperRequest.js');
let Action_action_parameters = require('./Action_action_parameters.js');
let Orientation = require('./Orientation.js');
let ConstrainedJointAngles = require('./ConstrainedJointAngles.js');
let JointSpeed = require('./JointSpeed.js');
let ControllerType = require('./ControllerType.js');
let SequenceList = require('./SequenceList.js');
let ChangeTwist = require('./ChangeTwist.js');
let RobotEventNotification = require('./RobotEventNotification.js');
let ControllerBehavior = require('./ControllerBehavior.js');
let TransformationRow = require('./TransformationRow.js');
let MapEvent_events = require('./MapEvent_events.js');
let ServoingMode = require('./ServoingMode.js');
let Base_GpioConfiguration = require('./Base_GpioConfiguration.js');
let ControllerNotification_state = require('./ControllerNotification_state.js');
let MapGroupHandle = require('./MapGroupHandle.js');
let MappingInfoNotification = require('./MappingInfoNotification.js');
let TransformationMatrix = require('./TransformationMatrix.js');
let JointsLimitationsList = require('./JointsLimitationsList.js');
let Point = require('./Point.js');
let JointAngles = require('./JointAngles.js');
let WrenchCommand = require('./WrenchCommand.js');
let ConstrainedOrientation = require('./ConstrainedOrientation.js');
let SequenceHandle = require('./SequenceHandle.js');
let WaypointValidationReport = require('./WaypointValidationReport.js');
let JointTorques = require('./JointTorques.js');
let FullUserProfile = require('./FullUserProfile.js');
let BackupEvent = require('./BackupEvent.js');
let IPv4Information = require('./IPv4Information.js');
let ControllerNotification = require('./ControllerNotification.js');
let ConfigurationNotificationEvent = require('./ConfigurationNotificationEvent.js');
let OperatingModeNotificationList = require('./OperatingModeNotificationList.js');
let JointAngle = require('./JointAngle.js');
let OperatingMode = require('./OperatingMode.js');
let SequenceTasksConfiguration = require('./SequenceTasksConfiguration.js');
let NavigationDirection = require('./NavigationDirection.js');
let ControlModeNotificationList = require('./ControlModeNotificationList.js');
let PasswordChange = require('./PasswordChange.js');
let IPv4Configuration = require('./IPv4Configuration.js');
let Delay = require('./Delay.js');
let SequenceTasksPair = require('./SequenceTasksPair.js');
let WifiEnableState = require('./WifiEnableState.js');
let CartesianLimitationList = require('./CartesianLimitationList.js');
let MapList = require('./MapList.js');
let GripperCommand = require('./GripperCommand.js');
let Finger = require('./Finger.js');
let WrenchLimitation = require('./WrenchLimitation.js');
let CartesianTrajectoryConstraint = require('./CartesianTrajectoryConstraint.js');
let Snapshot = require('./Snapshot.js');
let EmergencyStop = require('./EmergencyStop.js');
let ControllerHandle = require('./ControllerHandle.js');
let ActionEvent = require('./ActionEvent.js');
let GpioPinPropertyFlags = require('./GpioPinPropertyFlags.js');
let ChangeWrench = require('./ChangeWrench.js');
let SequenceTaskConfiguration = require('./SequenceTaskConfiguration.js');
let ControllerConfigurationMode = require('./ControllerConfigurationMode.js');
let TrajectoryErrorElement = require('./TrajectoryErrorElement.js');
let MappingInfoNotificationList = require('./MappingInfoNotificationList.js');
let TwistCommand = require('./TwistCommand.js');
let ControllerConfiguration = require('./ControllerConfiguration.js');
let ConfigurationChangeNotification_configuration_change = require('./ConfigurationChangeNotification_configuration_change.js');
let Xbox360DigitalInputIdentifier = require('./Xbox360DigitalInputIdentifier.js');
let Sequence = require('./Sequence.js');
let TrajectoryErrorIdentifier = require('./TrajectoryErrorIdentifier.js');
let MapGroup = require('./MapGroup.js');
let JointTrajectoryConstraintType = require('./JointTrajectoryConstraintType.js');
let ProtectionZoneNotification = require('./ProtectionZoneNotification.js');
let AdvancedSequenceHandle = require('./AdvancedSequenceHandle.js');
let WifiConfigurationList = require('./WifiConfigurationList.js');
let ActionExecutionState = require('./ActionExecutionState.js');
let MapElement = require('./MapElement.js');
let ControllerEventType = require('./ControllerEventType.js');
let CommunicationInterfaceConfiguration = require('./CommunicationInterfaceConfiguration.js');
let PreComputedJointTrajectory = require('./PreComputedJointTrajectory.js');
let ControllerElementState = require('./ControllerElementState.js');
let WristDigitalInputIdentifier = require('./WristDigitalInputIdentifier.js');
let SnapshotType = require('./SnapshotType.js');
let JointTorque = require('./JointTorque.js');
let MapHandle = require('./MapHandle.js');
let IKData = require('./IKData.js');
let FirmwareComponentVersion = require('./FirmwareComponentVersion.js');
let GpioEvent = require('./GpioEvent.js');
let UserProfile = require('./UserProfile.js');
let Base_RotationMatrixRow = require('./Base_RotationMatrixRow.js');
let SystemTime = require('./SystemTime.js');
let ProtectionZoneNotificationList = require('./ProtectionZoneNotificationList.js');
let ArmStateNotification = require('./ArmStateNotification.js');
let Gripper = require('./Gripper.js');
let ProtectionZoneInformation = require('./ProtectionZoneInformation.js');
let ConstrainedPose = require('./ConstrainedPose.js');
let TwistLimitation = require('./TwistLimitation.js');
let SequenceTasksRange = require('./SequenceTasksRange.js');
let BridgeType = require('./BridgeType.js');
let Admittance = require('./Admittance.js');
let SequenceInfoNotification = require('./SequenceInfoNotification.js');
let ActuatorInformation = require('./ActuatorInformation.js');
let ControllerElementHandle_identifier = require('./ControllerElementHandle_identifier.js');
let EventIdSequenceInfoNotification = require('./EventIdSequenceInfoNotification.js');
let ActivateMapHandle = require('./ActivateMapHandle.js');
let Waypoint = require('./Waypoint.js');
let WrenchMode = require('./WrenchMode.js');
let NetworkEvent = require('./NetworkEvent.js');
let SafetyEvent = require('./SafetyEvent.js');
let ActionNotificationList = require('./ActionNotificationList.js');
let ProtectionZoneEvent = require('./ProtectionZoneEvent.js');
let ControllerNotificationList = require('./ControllerNotificationList.js');
let SignalQuality = require('./SignalQuality.js');
let Waypoint_type_of_waypoint = require('./Waypoint_type_of_waypoint.js');
let BridgeStatus = require('./BridgeStatus.js');
let UserNotification = require('./UserNotification.js');
let GpioCommand = require('./GpioCommand.js');
let NetworkHandle = require('./NetworkHandle.js');
let ControllerConfigurationList = require('./ControllerConfigurationList.js');
let BridgeResult = require('./BridgeResult.js');
let ProtectionZoneHandle = require('./ProtectionZoneHandle.js');
let ProtectionZone = require('./ProtectionZone.js');
let GpioBehavior = require('./GpioBehavior.js');
let ActionHandle = require('./ActionHandle.js');
let SequenceTaskHandle = require('./SequenceTaskHandle.js');
let ArmStateInformation = require('./ArmStateInformation.js');
let ConstrainedJointAngle = require('./ConstrainedJointAngle.js');
let NetworkNotificationList = require('./NetworkNotificationList.js');
let FirmwareBundleVersions = require('./FirmwareBundleVersions.js');
let WifiEncryptionType = require('./WifiEncryptionType.js');
let SequenceTask = require('./SequenceTask.js');
let WifiInformationList = require('./WifiInformationList.js');
let UserProfileList = require('./UserProfileList.js');
let BluetoothEnableState = require('./BluetoothEnableState.js');
let Base_JointSpeeds = require('./Base_JointSpeeds.js');
let Base_RotationMatrix = require('./Base_RotationMatrix.js');
let ControllerEvent = require('./ControllerEvent.js');
let Base_CapSenseMode = require('./Base_CapSenseMode.js');
let SoundType = require('./SoundType.js');
let OperatingModeNotification = require('./OperatingModeNotification.js');
let TrajectoryContinuityMode = require('./TrajectoryContinuityMode.js');
let TrajectoryInfo = require('./TrajectoryInfo.js');
let Base_ControlMode = require('./Base_ControlMode.js');
let ServoingModeNotificationList = require('./ServoingModeNotificationList.js');
let TrajectoryErrorType = require('./TrajectoryErrorType.js');
let Twist = require('./Twist.js');
let SwitchControlMapping = require('./SwitchControlMapping.js');
let UserList = require('./UserList.js');
let UserEvent = require('./UserEvent.js');
let CartesianSpeed = require('./CartesianSpeed.js');
let AdmittanceMode = require('./AdmittanceMode.js');
let Xbox360AnalogInputIdentifier = require('./Xbox360AnalogInputIdentifier.js');
let LimitationType = require('./LimitationType.js');
let ControllerElementHandle = require('./ControllerElementHandle.js');
let Action = require('./Action.js');
let ControllerList = require('./ControllerList.js');
let Gen3GpioPinId = require('./Gen3GpioPinId.js');
let Base_CapSenseConfig = require('./Base_CapSenseConfig.js');
let LedState = require('./LedState.js');
let ControllerElementEventType = require('./ControllerElementEventType.js');
let Mapping = require('./Mapping.js');
let AngularWaypoint = require('./AngularWaypoint.js');
let UserNotificationList = require('./UserNotificationList.js');
let ActionNotification = require('./ActionNotification.js');
let PreComputedJointTrajectoryElement = require('./PreComputedJointTrajectoryElement.js');
let WifiConfiguration = require('./WifiConfiguration.js');
let ConstrainedPosition = require('./ConstrainedPosition.js');
let SequenceInformation = require('./SequenceInformation.js');
let GpioConfigurationList = require('./GpioConfigurationList.js');
let ControllerInputType = require('./ControllerInputType.js');
let GpioPinConfiguration = require('./GpioPinConfiguration.js');
let Base_ControlModeInformation = require('./Base_ControlModeInformation.js');
let FactoryNotification = require('./FactoryNotification.js');
let Base_Stop = require('./Base_Stop.js');
let OperatingModeInformation = require('./OperatingModeInformation.js');
let JointLimitation = require('./JointLimitation.js');
let NetworkType = require('./NetworkType.js');
let WifiSecurityType = require('./WifiSecurityType.js');
let GpioAction = require('./GpioAction.js');
let FactoryEvent = require('./FactoryEvent.js');
let ActionList = require('./ActionList.js');
let TrajectoryErrorReport = require('./TrajectoryErrorReport.js');
let SequenceInfoNotificationList = require('./SequenceInfoNotificationList.js');
let RequestedActionType = require('./RequestedActionType.js');
let BridgePortConfig = require('./BridgePortConfig.js');
let MappingHandle = require('./MappingHandle.js');
let ServoingModeNotification = require('./ServoingModeNotification.js');
let RFConfiguration = require('./RFConfiguration.js');
let BridgeConfig = require('./BridgeConfig.js');
let GripperMode = require('./GripperMode.js');
let BridgeList = require('./BridgeList.js');
let ZoneShape = require('./ZoneShape.js');
let SafetyNotificationList = require('./SafetyNotificationList.js');
let ActionType = require('./ActionType.js');
let MappingList = require('./MappingList.js');
let ChangeJointSpeeds = require('./ChangeJointSpeeds.js');
let Base_Position = require('./Base_Position.js');
let ConfigurationChangeNotification = require('./ConfigurationChangeNotification.js');
let Timeout = require('./Timeout.js');
let FullIPv4Configuration = require('./FullIPv4Configuration.js');
let Base_ControlModeNotification = require('./Base_ControlModeNotification.js');
let Faults = require('./Faults.js');
let Base_SafetyIdentifier = require('./Base_SafetyIdentifier.js');
let Wrench = require('./Wrench.js');
let NetworkNotification = require('./NetworkNotification.js');
let WifiInformation = require('./WifiInformation.js');
let ActuatorCustomData = require('./ActuatorCustomData.js');
let BaseFeedback = require('./BaseFeedback.js');
let ActuatorFeedback = require('./ActuatorFeedback.js');
let BaseCyclic_CustomData = require('./BaseCyclic_CustomData.js');
let BaseCyclic_Feedback = require('./BaseCyclic_Feedback.js');
let BaseCyclic_ServiceVersion = require('./BaseCyclic_ServiceVersion.js');
let BaseCyclic_Command = require('./BaseCyclic_Command.js');
let ActuatorCommand = require('./ActuatorCommand.js');
let ArmState = require('./ArmState.js');
let Connection = require('./Connection.js');
let UARTSpeed = require('./UARTSpeed.js');
let UARTParity = require('./UARTParity.js');
let UARTDeviceIdentification = require('./UARTDeviceIdentification.js');
let Empty = require('./Empty.js');
let CountryCodeIdentifier = require('./CountryCodeIdentifier.js');
let Timestamp = require('./Timestamp.js');
let UARTWordLength = require('./UARTWordLength.js');
let SafetyNotification = require('./SafetyNotification.js');
let UserProfileHandle = require('./UserProfileHandle.js');
let UARTStopBits = require('./UARTStopBits.js');
let UARTConfiguration = require('./UARTConfiguration.js');
let CartesianReferenceFrame = require('./CartesianReferenceFrame.js');
let DeviceTypes = require('./DeviceTypes.js');
let SafetyHandle = require('./SafetyHandle.js');
let NotificationHandle = require('./NotificationHandle.js');
let CountryCode = require('./CountryCode.js');
let NotificationOptions = require('./NotificationOptions.js');
let Permission = require('./Permission.js');
let SafetyStatusValue = require('./SafetyStatusValue.js');
let NotificationType = require('./NotificationType.js');
let DeviceHandle = require('./DeviceHandle.js');
let Unit = require('./Unit.js');
let ToolConfiguration = require('./ToolConfiguration.js');
let ControlConfig_ControlModeNotification = require('./ControlConfig_ControlModeNotification.js');
let ControlConfigurationNotification = require('./ControlConfigurationNotification.js');
let GravityVector = require('./GravityVector.js');
let JointAccelerationSoftLimits = require('./JointAccelerationSoftLimits.js');
let ControlConfig_ControlMode = require('./ControlConfig_ControlMode.js');
let TwistLinearSoftLimit = require('./TwistLinearSoftLimit.js');
let JointSpeedSoftLimits = require('./JointSpeedSoftLimits.js');
let KinematicLimitsList = require('./KinematicLimitsList.js');
let PayloadInformation = require('./PayloadInformation.js');
let KinematicLimits = require('./KinematicLimits.js');
let CartesianTransform = require('./CartesianTransform.js');
let AngularTwist = require('./AngularTwist.js');
let CartesianReferenceFrameInfo = require('./CartesianReferenceFrameInfo.js');
let ControlConfig_Position = require('./ControlConfig_Position.js');
let LinearTwist = require('./LinearTwist.js');
let ControlConfigurationEvent = require('./ControlConfigurationEvent.js');
let TwistAngularSoftLimit = require('./TwistAngularSoftLimit.js');
let ControlConfig_ControlModeInformation = require('./ControlConfig_ControlModeInformation.js');
let ControlConfig_ServiceVersion = require('./ControlConfig_ServiceVersion.js');
let DesiredSpeeds = require('./DesiredSpeeds.js');
let ControlConfig_JointSpeeds = require('./ControlConfig_JointSpeeds.js');
let FirmwareVersion = require('./FirmwareVersion.js');
let RunMode = require('./RunMode.js');
let DeviceConfig_CapSenseMode = require('./DeviceConfig_CapSenseMode.js');
let MACAddress = require('./MACAddress.js');
let DeviceConfig_SafetyLimitType = require('./DeviceConfig_SafetyLimitType.js');
let PowerOnSelfTestResult = require('./PowerOnSelfTestResult.js');
let SafetyThreshold = require('./SafetyThreshold.js');
let CalibrationParameter = require('./CalibrationParameter.js');
let SerialNumber = require('./SerialNumber.js');
let SafetyConfigurationList = require('./SafetyConfigurationList.js');
let RebootRqst = require('./RebootRqst.js');
let CalibrationResult = require('./CalibrationResult.js');
let SafetyInformationList = require('./SafetyInformationList.js');
let Calibration = require('./Calibration.js');
let DeviceType = require('./DeviceType.js');
let CalibrationElement = require('./CalibrationElement.js');
let CapSenseRegister = require('./CapSenseRegister.js');
let DeviceConfig_CapSenseConfig = require('./DeviceConfig_CapSenseConfig.js');
let SafetyInformation = require('./SafetyInformation.js');
let PartNumberRevision = require('./PartNumberRevision.js');
let PartNumber = require('./PartNumber.js');
let CalibrationStatus = require('./CalibrationStatus.js');
let CalibrationItem = require('./CalibrationItem.js');
let ModelNumber = require('./ModelNumber.js');
let IPv4Settings = require('./IPv4Settings.js');
let RunModes = require('./RunModes.js');
let DeviceConfig_ServiceVersion = require('./DeviceConfig_ServiceVersion.js');
let SafetyStatus = require('./SafetyStatus.js');
let BootloaderVersion = require('./BootloaderVersion.js');
let SafetyConfiguration = require('./SafetyConfiguration.js');
let CalibrationParameter_value = require('./CalibrationParameter_value.js');
let SafetyEnable = require('./SafetyEnable.js');
let DeviceManager_ServiceVersion = require('./DeviceManager_ServiceVersion.js');
let DeviceHandles = require('./DeviceHandles.js');
let RobotiqGripperStatusFlags = require('./RobotiqGripperStatusFlags.js');
let GripperConfig_SafetyIdentifier = require('./GripperConfig_SafetyIdentifier.js');
let GripperCyclic_ServiceVersion = require('./GripperCyclic_ServiceVersion.js');
let GripperCyclic_MessageId = require('./GripperCyclic_MessageId.js');
let MotorCommand = require('./MotorCommand.js');
let MotorFeedback = require('./MotorFeedback.js');
let GripperCyclic_CustomData = require('./GripperCyclic_CustomData.js');
let GripperCyclic_Feedback = require('./GripperCyclic_Feedback.js');
let CustomDataUnit = require('./CustomDataUnit.js');
let GripperCyclic_Command = require('./GripperCyclic_Command.js');
let InterconnectConfig_GPIOConfiguration = require('./InterconnectConfig_GPIOConfiguration.js');
let I2CMode = require('./I2CMode.js');
let InterconnectConfig_ServiceVersion = require('./InterconnectConfig_ServiceVersion.js');
let GPIOIdentification = require('./GPIOIdentification.js');
let I2CDeviceIdentification = require('./I2CDeviceIdentification.js');
let EthernetDeviceIdentification = require('./EthernetDeviceIdentification.js');
let I2CWriteRegisterParameter = require('./I2CWriteRegisterParameter.js');
let UARTPortId = require('./UARTPortId.js');
let GPIOPull = require('./GPIOPull.js');
let InterconnectConfig_SafetyIdentifier = require('./InterconnectConfig_SafetyIdentifier.js');
let GPIOIdentifier = require('./GPIOIdentifier.js');
let I2CDevice = require('./I2CDevice.js');
let EthernetConfiguration = require('./EthernetConfiguration.js');
let GPIOState = require('./GPIOState.js');
let EthernetSpeed = require('./EthernetSpeed.js');
let EthernetDevice = require('./EthernetDevice.js');
let I2CRegisterAddressSize = require('./I2CRegisterAddressSize.js');
let I2CConfiguration = require('./I2CConfiguration.js');
let I2CReadRegisterParameter = require('./I2CReadRegisterParameter.js');
let EthernetDuplex = require('./EthernetDuplex.js');
let I2CDeviceAddressing = require('./I2CDeviceAddressing.js');
let I2CReadParameter = require('./I2CReadParameter.js');
let GPIOMode = require('./GPIOMode.js');
let GPIOValue = require('./GPIOValue.js');
let I2CWriteParameter = require('./I2CWriteParameter.js');
let I2CData = require('./I2CData.js');
let InterconnectCyclic_CustomData = require('./InterconnectCyclic_CustomData.js');
let InterconnectCyclic_Feedback = require('./InterconnectCyclic_Feedback.js');
let InterconnectCyclic_Command_tool_command = require('./InterconnectCyclic_Command_tool_command.js');
let InterconnectCyclic_ServiceVersion = require('./InterconnectCyclic_ServiceVersion.js');
let InterconnectCyclic_Feedback_tool_feedback = require('./InterconnectCyclic_Feedback_tool_feedback.js');
let InterconnectCyclic_MessageId = require('./InterconnectCyclic_MessageId.js');
let InterconnectCyclic_Command = require('./InterconnectCyclic_Command.js');
let InterconnectCyclic_CustomData_tool_customData = require('./InterconnectCyclic_CustomData_tool_customData.js');
let InterfaceModuleType = require('./InterfaceModuleType.js');
let CompleteProductConfiguration = require('./CompleteProductConfiguration.js');
let BaseType = require('./BaseType.js');
let WristType = require('./WristType.js');
let ModelId = require('./ModelId.js');
let EndEffectorType = require('./EndEffectorType.js');
let ArmLaterality = require('./ArmLaterality.js');
let VisionModuleType = require('./VisionModuleType.js');
let ProductConfigurationEndEffectorType = require('./ProductConfigurationEndEffectorType.js');
let BrakeType = require('./BrakeType.js');
let FocusAction = require('./FocusAction.js');
let OptionValue = require('./OptionValue.js');
let FocusPoint = require('./FocusPoint.js');
let VisionConfig_RotationMatrix = require('./VisionConfig_RotationMatrix.js');
let VisionConfig_RotationMatrixRow = require('./VisionConfig_RotationMatrixRow.js');
let Option = require('./Option.js');
let DistortionCoefficients = require('./DistortionCoefficients.js');
let VisionNotification = require('./VisionNotification.js');
let ExtrinsicParameters = require('./ExtrinsicParameters.js');
let Resolution = require('./Resolution.js');
let TranslationVector = require('./TranslationVector.js');
let SensorIdentifier = require('./SensorIdentifier.js');
let IntrinsicParameters = require('./IntrinsicParameters.js');
let OptionInformation = require('./OptionInformation.js');
let ManualFocus = require('./ManualFocus.js');
let VisionConfig_ServiceVersion = require('./VisionConfig_ServiceVersion.js');
let FrameRate = require('./FrameRate.js');
let VisionEvent = require('./VisionEvent.js');
let SensorFocusAction_action_parameters = require('./SensorFocusAction_action_parameters.js');
let SensorFocusAction = require('./SensorFocusAction.js');
let Sensor = require('./Sensor.js');
let SensorSettings = require('./SensorSettings.js');
let OptionIdentifier = require('./OptionIdentifier.js');
let IntrinsicProfileIdentifier = require('./IntrinsicProfileIdentifier.js');
let BitRate = require('./BitRate.js');
let FollowCartesianTrajectoryActionGoal = require('./FollowCartesianTrajectoryActionGoal.js');
let FollowCartesianTrajectoryResult = require('./FollowCartesianTrajectoryResult.js');
let FollowCartesianTrajectoryGoal = require('./FollowCartesianTrajectoryGoal.js');
let FollowCartesianTrajectoryActionFeedback = require('./FollowCartesianTrajectoryActionFeedback.js');
let FollowCartesianTrajectoryFeedback = require('./FollowCartesianTrajectoryFeedback.js');
let FollowCartesianTrajectoryActionResult = require('./FollowCartesianTrajectoryActionResult.js');
let FollowCartesianTrajectoryAction = require('./FollowCartesianTrajectoryAction.js');

module.exports = {
  ApiOptions: ApiOptions,
  KortexError: KortexError,
  ErrorCodes: ErrorCodes,
  SubErrorCodes: SubErrorCodes,
  CoggingFeedforwardModeInformation: CoggingFeedforwardModeInformation,
  ActuatorConfig_ServiceVersion: ActuatorConfig_ServiceVersion,
  EncoderDerivativeParameters: EncoderDerivativeParameters,
  CoggingFeedforwardMode: CoggingFeedforwardMode,
  FrequencyResponse: FrequencyResponse,
  CustomDataSelection: CustomDataSelection,
  ControlLoopSelection: ControlLoopSelection,
  ControlLoop: ControlLoop,
  VectorDriveParameters: VectorDriveParameters,
  CommandMode: CommandMode,
  StepResponse: StepResponse,
  AxisPosition: AxisPosition,
  Servoing: Servoing,
  CommandModeInformation: CommandModeInformation,
  SafetyIdentifierBankA: SafetyIdentifierBankA,
  RampResponse: RampResponse,
  PositionCommand: PositionCommand,
  ControlLoopParameters: ControlLoopParameters,
  TorqueCalibration: TorqueCalibration,
  AxisOffsets: AxisOffsets,
  CustomDataIndex: CustomDataIndex,
  ActuatorConfig_SafetyLimitType: ActuatorConfig_SafetyLimitType,
  ActuatorConfig_ControlModeInformation: ActuatorConfig_ControlModeInformation,
  LoopSelection: LoopSelection,
  TorqueOffset: TorqueOffset,
  ActuatorConfig_ControlMode: ActuatorConfig_ControlMode,
  ActuatorCyclic_Command: ActuatorCyclic_Command,
  CommandFlags: CommandFlags,
  ActuatorCyclic_Feedback: ActuatorCyclic_Feedback,
  ActuatorCyclic_CustomData: ActuatorCyclic_CustomData,
  StatusFlags: StatusFlags,
  ActuatorCyclic_MessageId: ActuatorCyclic_MessageId,
  ActuatorCyclic_ServiceVersion: ActuatorCyclic_ServiceVersion,
  RobotEvent: RobotEvent,
  MapEvent: MapEvent,
  CartesianTrajectoryConstraint_type: CartesianTrajectoryConstraint_type,
  Base_ServiceVersion: Base_ServiceVersion,
  Ssid: Ssid,
  CartesianLimitation: CartesianLimitation,
  BridgeIdentifier: BridgeIdentifier,
  Query: Query,
  ServoingModeInformation: ServoingModeInformation,
  ProtectionZoneList: ProtectionZoneList,
  Pose: Pose,
  AppendActionInformation: AppendActionInformation,
  JointNavigationDirection: JointNavigationDirection,
  MapGroupList: MapGroupList,
  KinematicTrajectoryConstraints: KinematicTrajectoryConstraints,
  Map: Map,
  JointTrajectoryConstraint: JointTrajectoryConstraint,
  WaypointList: WaypointList,
  RobotEventNotificationList: RobotEventNotificationList,
  ShapeType: ShapeType,
  ConfigurationChangeNotificationList: ConfigurationChangeNotificationList,
  ControllerState: ControllerState,
  TrajectoryInfoType: TrajectoryInfoType,
  SequenceTasks: SequenceTasks,
  CartesianWaypoint: CartesianWaypoint,
  GripperRequest: GripperRequest,
  Action_action_parameters: Action_action_parameters,
  Orientation: Orientation,
  ConstrainedJointAngles: ConstrainedJointAngles,
  JointSpeed: JointSpeed,
  ControllerType: ControllerType,
  SequenceList: SequenceList,
  ChangeTwist: ChangeTwist,
  RobotEventNotification: RobotEventNotification,
  ControllerBehavior: ControllerBehavior,
  TransformationRow: TransformationRow,
  MapEvent_events: MapEvent_events,
  ServoingMode: ServoingMode,
  Base_GpioConfiguration: Base_GpioConfiguration,
  ControllerNotification_state: ControllerNotification_state,
  MapGroupHandle: MapGroupHandle,
  MappingInfoNotification: MappingInfoNotification,
  TransformationMatrix: TransformationMatrix,
  JointsLimitationsList: JointsLimitationsList,
  Point: Point,
  JointAngles: JointAngles,
  WrenchCommand: WrenchCommand,
  ConstrainedOrientation: ConstrainedOrientation,
  SequenceHandle: SequenceHandle,
  WaypointValidationReport: WaypointValidationReport,
  JointTorques: JointTorques,
  FullUserProfile: FullUserProfile,
  BackupEvent: BackupEvent,
  IPv4Information: IPv4Information,
  ControllerNotification: ControllerNotification,
  ConfigurationNotificationEvent: ConfigurationNotificationEvent,
  OperatingModeNotificationList: OperatingModeNotificationList,
  JointAngle: JointAngle,
  OperatingMode: OperatingMode,
  SequenceTasksConfiguration: SequenceTasksConfiguration,
  NavigationDirection: NavigationDirection,
  ControlModeNotificationList: ControlModeNotificationList,
  PasswordChange: PasswordChange,
  IPv4Configuration: IPv4Configuration,
  Delay: Delay,
  SequenceTasksPair: SequenceTasksPair,
  WifiEnableState: WifiEnableState,
  CartesianLimitationList: CartesianLimitationList,
  MapList: MapList,
  GripperCommand: GripperCommand,
  Finger: Finger,
  WrenchLimitation: WrenchLimitation,
  CartesianTrajectoryConstraint: CartesianTrajectoryConstraint,
  Snapshot: Snapshot,
  EmergencyStop: EmergencyStop,
  ControllerHandle: ControllerHandle,
  ActionEvent: ActionEvent,
  GpioPinPropertyFlags: GpioPinPropertyFlags,
  ChangeWrench: ChangeWrench,
  SequenceTaskConfiguration: SequenceTaskConfiguration,
  ControllerConfigurationMode: ControllerConfigurationMode,
  TrajectoryErrorElement: TrajectoryErrorElement,
  MappingInfoNotificationList: MappingInfoNotificationList,
  TwistCommand: TwistCommand,
  ControllerConfiguration: ControllerConfiguration,
  ConfigurationChangeNotification_configuration_change: ConfigurationChangeNotification_configuration_change,
  Xbox360DigitalInputIdentifier: Xbox360DigitalInputIdentifier,
  Sequence: Sequence,
  TrajectoryErrorIdentifier: TrajectoryErrorIdentifier,
  MapGroup: MapGroup,
  JointTrajectoryConstraintType: JointTrajectoryConstraintType,
  ProtectionZoneNotification: ProtectionZoneNotification,
  AdvancedSequenceHandle: AdvancedSequenceHandle,
  WifiConfigurationList: WifiConfigurationList,
  ActionExecutionState: ActionExecutionState,
  MapElement: MapElement,
  ControllerEventType: ControllerEventType,
  CommunicationInterfaceConfiguration: CommunicationInterfaceConfiguration,
  PreComputedJointTrajectory: PreComputedJointTrajectory,
  ControllerElementState: ControllerElementState,
  WristDigitalInputIdentifier: WristDigitalInputIdentifier,
  SnapshotType: SnapshotType,
  JointTorque: JointTorque,
  MapHandle: MapHandle,
  IKData: IKData,
  FirmwareComponentVersion: FirmwareComponentVersion,
  GpioEvent: GpioEvent,
  UserProfile: UserProfile,
  Base_RotationMatrixRow: Base_RotationMatrixRow,
  SystemTime: SystemTime,
  ProtectionZoneNotificationList: ProtectionZoneNotificationList,
  ArmStateNotification: ArmStateNotification,
  Gripper: Gripper,
  ProtectionZoneInformation: ProtectionZoneInformation,
  ConstrainedPose: ConstrainedPose,
  TwistLimitation: TwistLimitation,
  SequenceTasksRange: SequenceTasksRange,
  BridgeType: BridgeType,
  Admittance: Admittance,
  SequenceInfoNotification: SequenceInfoNotification,
  ActuatorInformation: ActuatorInformation,
  ControllerElementHandle_identifier: ControllerElementHandle_identifier,
  EventIdSequenceInfoNotification: EventIdSequenceInfoNotification,
  ActivateMapHandle: ActivateMapHandle,
  Waypoint: Waypoint,
  WrenchMode: WrenchMode,
  NetworkEvent: NetworkEvent,
  SafetyEvent: SafetyEvent,
  ActionNotificationList: ActionNotificationList,
  ProtectionZoneEvent: ProtectionZoneEvent,
  ControllerNotificationList: ControllerNotificationList,
  SignalQuality: SignalQuality,
  Waypoint_type_of_waypoint: Waypoint_type_of_waypoint,
  BridgeStatus: BridgeStatus,
  UserNotification: UserNotification,
  GpioCommand: GpioCommand,
  NetworkHandle: NetworkHandle,
  ControllerConfigurationList: ControllerConfigurationList,
  BridgeResult: BridgeResult,
  ProtectionZoneHandle: ProtectionZoneHandle,
  ProtectionZone: ProtectionZone,
  GpioBehavior: GpioBehavior,
  ActionHandle: ActionHandle,
  SequenceTaskHandle: SequenceTaskHandle,
  ArmStateInformation: ArmStateInformation,
  ConstrainedJointAngle: ConstrainedJointAngle,
  NetworkNotificationList: NetworkNotificationList,
  FirmwareBundleVersions: FirmwareBundleVersions,
  WifiEncryptionType: WifiEncryptionType,
  SequenceTask: SequenceTask,
  WifiInformationList: WifiInformationList,
  UserProfileList: UserProfileList,
  BluetoothEnableState: BluetoothEnableState,
  Base_JointSpeeds: Base_JointSpeeds,
  Base_RotationMatrix: Base_RotationMatrix,
  ControllerEvent: ControllerEvent,
  Base_CapSenseMode: Base_CapSenseMode,
  SoundType: SoundType,
  OperatingModeNotification: OperatingModeNotification,
  TrajectoryContinuityMode: TrajectoryContinuityMode,
  TrajectoryInfo: TrajectoryInfo,
  Base_ControlMode: Base_ControlMode,
  ServoingModeNotificationList: ServoingModeNotificationList,
  TrajectoryErrorType: TrajectoryErrorType,
  Twist: Twist,
  SwitchControlMapping: SwitchControlMapping,
  UserList: UserList,
  UserEvent: UserEvent,
  CartesianSpeed: CartesianSpeed,
  AdmittanceMode: AdmittanceMode,
  Xbox360AnalogInputIdentifier: Xbox360AnalogInputIdentifier,
  LimitationType: LimitationType,
  ControllerElementHandle: ControllerElementHandle,
  Action: Action,
  ControllerList: ControllerList,
  Gen3GpioPinId: Gen3GpioPinId,
  Base_CapSenseConfig: Base_CapSenseConfig,
  LedState: LedState,
  ControllerElementEventType: ControllerElementEventType,
  Mapping: Mapping,
  AngularWaypoint: AngularWaypoint,
  UserNotificationList: UserNotificationList,
  ActionNotification: ActionNotification,
  PreComputedJointTrajectoryElement: PreComputedJointTrajectoryElement,
  WifiConfiguration: WifiConfiguration,
  ConstrainedPosition: ConstrainedPosition,
  SequenceInformation: SequenceInformation,
  GpioConfigurationList: GpioConfigurationList,
  ControllerInputType: ControllerInputType,
  GpioPinConfiguration: GpioPinConfiguration,
  Base_ControlModeInformation: Base_ControlModeInformation,
  FactoryNotification: FactoryNotification,
  Base_Stop: Base_Stop,
  OperatingModeInformation: OperatingModeInformation,
  JointLimitation: JointLimitation,
  NetworkType: NetworkType,
  WifiSecurityType: WifiSecurityType,
  GpioAction: GpioAction,
  FactoryEvent: FactoryEvent,
  ActionList: ActionList,
  TrajectoryErrorReport: TrajectoryErrorReport,
  SequenceInfoNotificationList: SequenceInfoNotificationList,
  RequestedActionType: RequestedActionType,
  BridgePortConfig: BridgePortConfig,
  MappingHandle: MappingHandle,
  ServoingModeNotification: ServoingModeNotification,
  RFConfiguration: RFConfiguration,
  BridgeConfig: BridgeConfig,
  GripperMode: GripperMode,
  BridgeList: BridgeList,
  ZoneShape: ZoneShape,
  SafetyNotificationList: SafetyNotificationList,
  ActionType: ActionType,
  MappingList: MappingList,
  ChangeJointSpeeds: ChangeJointSpeeds,
  Base_Position: Base_Position,
  ConfigurationChangeNotification: ConfigurationChangeNotification,
  Timeout: Timeout,
  FullIPv4Configuration: FullIPv4Configuration,
  Base_ControlModeNotification: Base_ControlModeNotification,
  Faults: Faults,
  Base_SafetyIdentifier: Base_SafetyIdentifier,
  Wrench: Wrench,
  NetworkNotification: NetworkNotification,
  WifiInformation: WifiInformation,
  ActuatorCustomData: ActuatorCustomData,
  BaseFeedback: BaseFeedback,
  ActuatorFeedback: ActuatorFeedback,
  BaseCyclic_CustomData: BaseCyclic_CustomData,
  BaseCyclic_Feedback: BaseCyclic_Feedback,
  BaseCyclic_ServiceVersion: BaseCyclic_ServiceVersion,
  BaseCyclic_Command: BaseCyclic_Command,
  ActuatorCommand: ActuatorCommand,
  ArmState: ArmState,
  Connection: Connection,
  UARTSpeed: UARTSpeed,
  UARTParity: UARTParity,
  UARTDeviceIdentification: UARTDeviceIdentification,
  Empty: Empty,
  CountryCodeIdentifier: CountryCodeIdentifier,
  Timestamp: Timestamp,
  UARTWordLength: UARTWordLength,
  SafetyNotification: SafetyNotification,
  UserProfileHandle: UserProfileHandle,
  UARTStopBits: UARTStopBits,
  UARTConfiguration: UARTConfiguration,
  CartesianReferenceFrame: CartesianReferenceFrame,
  DeviceTypes: DeviceTypes,
  SafetyHandle: SafetyHandle,
  NotificationHandle: NotificationHandle,
  CountryCode: CountryCode,
  NotificationOptions: NotificationOptions,
  Permission: Permission,
  SafetyStatusValue: SafetyStatusValue,
  NotificationType: NotificationType,
  DeviceHandle: DeviceHandle,
  Unit: Unit,
  ToolConfiguration: ToolConfiguration,
  ControlConfig_ControlModeNotification: ControlConfig_ControlModeNotification,
  ControlConfigurationNotification: ControlConfigurationNotification,
  GravityVector: GravityVector,
  JointAccelerationSoftLimits: JointAccelerationSoftLimits,
  ControlConfig_ControlMode: ControlConfig_ControlMode,
  TwistLinearSoftLimit: TwistLinearSoftLimit,
  JointSpeedSoftLimits: JointSpeedSoftLimits,
  KinematicLimitsList: KinematicLimitsList,
  PayloadInformation: PayloadInformation,
  KinematicLimits: KinematicLimits,
  CartesianTransform: CartesianTransform,
  AngularTwist: AngularTwist,
  CartesianReferenceFrameInfo: CartesianReferenceFrameInfo,
  ControlConfig_Position: ControlConfig_Position,
  LinearTwist: LinearTwist,
  ControlConfigurationEvent: ControlConfigurationEvent,
  TwistAngularSoftLimit: TwistAngularSoftLimit,
  ControlConfig_ControlModeInformation: ControlConfig_ControlModeInformation,
  ControlConfig_ServiceVersion: ControlConfig_ServiceVersion,
  DesiredSpeeds: DesiredSpeeds,
  ControlConfig_JointSpeeds: ControlConfig_JointSpeeds,
  FirmwareVersion: FirmwareVersion,
  RunMode: RunMode,
  DeviceConfig_CapSenseMode: DeviceConfig_CapSenseMode,
  MACAddress: MACAddress,
  DeviceConfig_SafetyLimitType: DeviceConfig_SafetyLimitType,
  PowerOnSelfTestResult: PowerOnSelfTestResult,
  SafetyThreshold: SafetyThreshold,
  CalibrationParameter: CalibrationParameter,
  SerialNumber: SerialNumber,
  SafetyConfigurationList: SafetyConfigurationList,
  RebootRqst: RebootRqst,
  CalibrationResult: CalibrationResult,
  SafetyInformationList: SafetyInformationList,
  Calibration: Calibration,
  DeviceType: DeviceType,
  CalibrationElement: CalibrationElement,
  CapSenseRegister: CapSenseRegister,
  DeviceConfig_CapSenseConfig: DeviceConfig_CapSenseConfig,
  SafetyInformation: SafetyInformation,
  PartNumberRevision: PartNumberRevision,
  PartNumber: PartNumber,
  CalibrationStatus: CalibrationStatus,
  CalibrationItem: CalibrationItem,
  ModelNumber: ModelNumber,
  IPv4Settings: IPv4Settings,
  RunModes: RunModes,
  DeviceConfig_ServiceVersion: DeviceConfig_ServiceVersion,
  SafetyStatus: SafetyStatus,
  BootloaderVersion: BootloaderVersion,
  SafetyConfiguration: SafetyConfiguration,
  CalibrationParameter_value: CalibrationParameter_value,
  SafetyEnable: SafetyEnable,
  DeviceManager_ServiceVersion: DeviceManager_ServiceVersion,
  DeviceHandles: DeviceHandles,
  RobotiqGripperStatusFlags: RobotiqGripperStatusFlags,
  GripperConfig_SafetyIdentifier: GripperConfig_SafetyIdentifier,
  GripperCyclic_ServiceVersion: GripperCyclic_ServiceVersion,
  GripperCyclic_MessageId: GripperCyclic_MessageId,
  MotorCommand: MotorCommand,
  MotorFeedback: MotorFeedback,
  GripperCyclic_CustomData: GripperCyclic_CustomData,
  GripperCyclic_Feedback: GripperCyclic_Feedback,
  CustomDataUnit: CustomDataUnit,
  GripperCyclic_Command: GripperCyclic_Command,
  InterconnectConfig_GPIOConfiguration: InterconnectConfig_GPIOConfiguration,
  I2CMode: I2CMode,
  InterconnectConfig_ServiceVersion: InterconnectConfig_ServiceVersion,
  GPIOIdentification: GPIOIdentification,
  I2CDeviceIdentification: I2CDeviceIdentification,
  EthernetDeviceIdentification: EthernetDeviceIdentification,
  I2CWriteRegisterParameter: I2CWriteRegisterParameter,
  UARTPortId: UARTPortId,
  GPIOPull: GPIOPull,
  InterconnectConfig_SafetyIdentifier: InterconnectConfig_SafetyIdentifier,
  GPIOIdentifier: GPIOIdentifier,
  I2CDevice: I2CDevice,
  EthernetConfiguration: EthernetConfiguration,
  GPIOState: GPIOState,
  EthernetSpeed: EthernetSpeed,
  EthernetDevice: EthernetDevice,
  I2CRegisterAddressSize: I2CRegisterAddressSize,
  I2CConfiguration: I2CConfiguration,
  I2CReadRegisterParameter: I2CReadRegisterParameter,
  EthernetDuplex: EthernetDuplex,
  I2CDeviceAddressing: I2CDeviceAddressing,
  I2CReadParameter: I2CReadParameter,
  GPIOMode: GPIOMode,
  GPIOValue: GPIOValue,
  I2CWriteParameter: I2CWriteParameter,
  I2CData: I2CData,
  InterconnectCyclic_CustomData: InterconnectCyclic_CustomData,
  InterconnectCyclic_Feedback: InterconnectCyclic_Feedback,
  InterconnectCyclic_Command_tool_command: InterconnectCyclic_Command_tool_command,
  InterconnectCyclic_ServiceVersion: InterconnectCyclic_ServiceVersion,
  InterconnectCyclic_Feedback_tool_feedback: InterconnectCyclic_Feedback_tool_feedback,
  InterconnectCyclic_MessageId: InterconnectCyclic_MessageId,
  InterconnectCyclic_Command: InterconnectCyclic_Command,
  InterconnectCyclic_CustomData_tool_customData: InterconnectCyclic_CustomData_tool_customData,
  InterfaceModuleType: InterfaceModuleType,
  CompleteProductConfiguration: CompleteProductConfiguration,
  BaseType: BaseType,
  WristType: WristType,
  ModelId: ModelId,
  EndEffectorType: EndEffectorType,
  ArmLaterality: ArmLaterality,
  VisionModuleType: VisionModuleType,
  ProductConfigurationEndEffectorType: ProductConfigurationEndEffectorType,
  BrakeType: BrakeType,
  FocusAction: FocusAction,
  OptionValue: OptionValue,
  FocusPoint: FocusPoint,
  VisionConfig_RotationMatrix: VisionConfig_RotationMatrix,
  VisionConfig_RotationMatrixRow: VisionConfig_RotationMatrixRow,
  Option: Option,
  DistortionCoefficients: DistortionCoefficients,
  VisionNotification: VisionNotification,
  ExtrinsicParameters: ExtrinsicParameters,
  Resolution: Resolution,
  TranslationVector: TranslationVector,
  SensorIdentifier: SensorIdentifier,
  IntrinsicParameters: IntrinsicParameters,
  OptionInformation: OptionInformation,
  ManualFocus: ManualFocus,
  VisionConfig_ServiceVersion: VisionConfig_ServiceVersion,
  FrameRate: FrameRate,
  VisionEvent: VisionEvent,
  SensorFocusAction_action_parameters: SensorFocusAction_action_parameters,
  SensorFocusAction: SensorFocusAction,
  Sensor: Sensor,
  SensorSettings: SensorSettings,
  OptionIdentifier: OptionIdentifier,
  IntrinsicProfileIdentifier: IntrinsicProfileIdentifier,
  BitRate: BitRate,
  FollowCartesianTrajectoryActionGoal: FollowCartesianTrajectoryActionGoal,
  FollowCartesianTrajectoryResult: FollowCartesianTrajectoryResult,
  FollowCartesianTrajectoryGoal: FollowCartesianTrajectoryGoal,
  FollowCartesianTrajectoryActionFeedback: FollowCartesianTrajectoryActionFeedback,
  FollowCartesianTrajectoryFeedback: FollowCartesianTrajectoryFeedback,
  FollowCartesianTrajectoryActionResult: FollowCartesianTrajectoryActionResult,
  FollowCartesianTrajectoryAction: FollowCartesianTrajectoryAction,
};
