/**  ____________________________________________________________________
 *
 *  SBGC32 Serial API Library v2.2.1
 *
 *  @file       serialAPI_Config.h
 *
 *  @brief      Configuration header file of the library
 *  ____________________________________________________________________
 */

#ifndef SERIAL_API_CONFIG_H_
#define SERIAL_API_CONFIG_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Disable and enable parameters */
#define sbgcOFF                 (0)
#define sbgcON                  (-1)

/* ---------------------------------------------------------
 *                      Build Options
 * --------------------------------------------------------- */

#define SBGC_NEED_SOURCES_MAKE  sbgcOFF     /* We compile sources ourselves */

/* ---------------------------------------------------------
 *                  Module Configuration
 * --------------------------------------------------------- */

#define SBGC_ADJVAR_MODULE      sbgcON
#define SBGC_CALIB_MODULE       sbgcOFF
#define SBGC_EEPROM_MODULE      sbgcOFF
#define SBGC_CONTROL_MODULE     sbgcON
#define SBGC_IMU_MODULE         sbgcOFF
#define SBGC_PROFILES_MODULE    sbgcON
#define SBGC_REALTIME_MODULE    sbgcON
#define SBGC_SERVICE_MODULE     sbgcON

/* ---------------------------------------------------------
 *              Main Serial API Configurations
 * --------------------------------------------------------- */

#define SBGC_SYS_BIG_ENDIAN     sbgcOFF
#define SBGC_SEVERAL_DEVICES    sbgcOFF
#define SBGC_PROTOCOL_VERSION   2

#define SBGC_NON_BLOCKING_MODE  sbgcOFF     /* keep it simple, blocking mode */

#define SBGC_TX_BUFFER_SIZE     1
#define SBGC_RX_BUFFER_SIZE     1
#define SBGC_RX_CMD_OLD_PRIOR   sbgcOFF

#define SBGC_NEED_DEBUG         sbgcON
#if (SBGC_NEED_DEBUG)
    #define SBGC_LOG_COMMAND_TIME   sbgcON
    #define SBGC_LOG_COMMAND_NUMBER sbgcON
    #define SBGC_LOG_COMMAND_DIR    sbgcON
    #define SBGC_LOG_COMMAND_NAME   sbgcON
    #define SBGC_LOG_COMMAND_ID     sbgcON
    #define SBGC_LOG_COMMAND_STATUS sbgcON
    #define SBGC_LOG_COMMAND_PARAM  sbgcON
    #define SBGC_LOG_COMMAND_DATA   sbgcOFF
    #define SBGC_CUSTOM_SPRINTF     sbgcOFF
#endif

#define SBGC_NEED_ASSERTS       sbgcOFF
#define SBGC_NEED_CONFIRM_CMD   sbgcON
#define SBGC_NEED_REF_INFO      sbgcOFF

#define SBGC_DEFAULT_TIMEOUT    100
#define SBGC_STARTUP_DELAY      500

#define ROLL                    0
#define PITCH                   1
#define YAW                     2

#if (SBGC_ADJVAR_MODULE)
    #define SBGC_ADJ_VARS_REF_INFO  sbgcON
    #define SBGC_ADJ_VARS_NAMES     sbgcON
    #define SBGC_ADJ_VARS_ADD_FLAG  sbgcOFF
#endif

#if (SBGC_NEED_DEBUG && SBGC_NEED_CONFIRM_CMD)
    #define SBGC_DETAILED_CONFIRM   sbgcOFF
#endif

/* ---------------------------------------------------------
 *            OS Support Configurations (all OFF)
 * --------------------------------------------------------- */

#define SBGC_USE_AZURE_RTOS     sbgcOFF
#define SBGC_USE_FREE_RTOS      sbgcOFF
#define SBGC_USE_PTHREAD_OS     sbgcOFF

/* ---------------------------------------------------------
 *                 Driver Configurations
 * --------------------------------------------------------- */

#define SBGC_USE_ARDUINO_DRIVER sbgcOFF
#define SBGC_USE_ESPIDF_DRIVER  sbgcOFF
#define SBGC_USE_LINUX_DRIVER   sbgcON
#define SBGC_USE_STM32_DRIVER   sbgcOFF

#if (SBGC_USE_LINUX_DRIVER && (SBGC_SEVERAL_DEVICES == sbgcOFF))
    /*  Make sure this matches your BaseCam USB device.   */
    #define SBGC_SERIAL_PORT    "/dev/ttyACM0"
    #define SBGC_SERIAL_SPEED   B115200
#endif

#ifdef __cplusplus
}
#endif

#endif /* SERIAL_API_CONFIG_H_ */
