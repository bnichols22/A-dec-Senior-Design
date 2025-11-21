#ifndef SERIALAPI_CONFIG_H_
#define SERIALAPI_CONFIG_H_

#define sbgcOFF  (0)
#define sbgcON   (-1)

/* ================================================================
   Build options
   ================================================================ */
#define SBGC_NEED_SOURCES_MAKE   sbgcON      // <<< REQUIRED for full static build
#define SBGC_NEED_DEBUG          sbgcON      // Debug prints OK

/* ================================================================
   Modules to include
   (Minimal working set that matches our functional build)
   ================================================================ */
#define SBGC_ADJVAR_MODULE       sbgcON
#define SBGC_CALIB_MODULE        sbgcOFF
#define SBGC_EEPROM_MODULE       sbgcOFF
#define SBGC_CONTROL_MODULE      sbgcON
#define SBGC_IMU_MODULE          sbgcOFF
#define SBGC_PROFILES_MODULE     sbgcON
#define SBGC_REALTIME_MODULE     sbgcON
#define SBGC_SERVICE_MODULE      sbgcON

/* ================================================================
   OS / Blocking Mode
   ================================================================ */
#define SBGC_NON_BLOCKING_MODE   sbgcOFF

#define SBGC_USE_AZURE_RTOS      sbgcOFF
#define SBGC_USE_FREE_RTOS       sbgcOFF
#define SBGC_USE_PTHREAD_OS      sbgcOFF

/* ================================================================
   Drivers
   ================================================================ */
#define SBGC_USE_ARDUINO_DRIVER  sbgcOFF
#define SBGC_USE_ESPIDF_DRIVER   sbgcOFF
#define SBGC_USE_LINUX_DRIVER    sbgcON     // <<< IMPORTANT : Linux driver ON
#define SBGC_USE_STM32_DRIVER    sbgcOFF

/* ================================================================
   Serial Port Definition
   (Matches the working sample from the repo)
   ================================================================ */
#define SBGC_SERIAL_PORT   "/dev/ttyUSB0"
#define SBGC_SERIAL_SPEED  B115200

#endif /* SERIALAPI_CONFIG_H_ */
