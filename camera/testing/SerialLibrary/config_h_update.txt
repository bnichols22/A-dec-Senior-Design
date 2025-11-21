/**	____________________________________________________________________
 *
 *	SBGC32 Serial API Library v2.2.1
 *
 *	@file		serialAPI_Config.h
 *
 *	@brief 		Configuration header file of the library (Pi + Linux driver)
 *	____________________________________________________________________
 */

#ifndef		SERIALAPI_CONFIG_H_
#define		SERIALAPI_CONFIG_H_

#ifdef		__cplusplus
extern		"C" {
#endif
/*  = = = = = = = = = = = = = = = = = = = = = = = */

/* Disable and enable parameters */
#define		sbgcOFF					(0)
#define		sbgcON					(-1)


/**	@addtogroup	Configurations
 *	@{
 */
/* ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
 *									   Build Options
 */
#define		SBGC_NEED_SOURCES_MAKE	sbgcON			/*!<  Collects all nested source files into the library's top level. Uses the
														  serialAPI_MakeCpp.cpp file to collect and compile C++ sources					*/


/* ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
 *								Module Configuration
 */
/* Link various library functions, divided into modules */
#define		SBGC_ADJVAR_MODULE		sbgcON			/*!<  Adjustable variables. See @ref Adjvar module									*/
#define		SBGC_CALIB_MODULE		sbgcOFF			/*!<  Calibration functions. See @ref Calib module									*/
#define		SBGC_EEPROM_MODULE		sbgcOFF			/*!<  EEPROM operations. See @ref EEPROM module										*/
#define		SBGC_CONTROL_MODULE		sbgcON			/*!<  Gimbal control. See @ref Gimbal_Control module								*/
#define		SBGC_IMU_MODULE			sbgcOFF			/*!<  IMU data functions. See @ref IMU module										*/
#define		SBGC_PROFILES_MODULE	sbgcON			/*!<  Profile configurations. See @ref Profiles module								*/
#define		SBGC_REALTIME_MODULE	sbgcON			/*!<  Realtime data processing. See @ref Realtime module							*/
#define		SBGC_SERVICE_MODULE		sbgcON			/*!<  Service functions. See @ref Service module									*/


/* ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
 *					  Main Serial API Configurations
 */
#define		SBGC_SYS_BIG_ENDIAN		sbgcOFF			/*!<  LITTLE ENDIAN (normal for Pi / x86)											*/
#define		SBGC_SEVERAL_DEVICES	sbgcOFF			/*!<  Using only one controller with the library									*/
#define		SBGC_PROTOCOL_VERSION	2				/*!<  V.1 or V.2 SerialAPI protocol version											*/

#define		SBGC_NON_BLOCKING_MODE	sbgcOFF			/*!<  Blocking API (simpler, matches demo)											*/
#if (SBGC_NON_BLOCKING_MODE)
	#define	SBGC_NEED_TOKENS		sbgcOFF
	#define	SBGC_NEED_CALLBACKS		sbgcOFF
	#define	SBGC_OPTIMIZE_COMMANDS	sbgcOFF
	#define	SBGC_SEND_IMMEDIATELY	sbgcOFF
	#define	SBGC_CHAINED_TIMEOUT	sbgcOFF
	#define	SBGC_MAX_COMMAND_NUM	2
#endif

#define		SBGC_TX_BUFFER_SIZE		1				/*!<  1 Min (256 bytes) --> 8 Max (32768 bytes). Buffer size for sent commands		*/
#define		SBGC_RX_BUFFER_SIZE		1				/*!<  1 Min (256 bytes) --> 8 Max (32768 bytes). Buffer size for received commands	*/
#define		SBGC_RX_CMD_OLD_PRIOR	sbgcOFF			/*!<  Don't drop new commands (default behavior)									*/

#define		SBGC_NEED_DEBUG			sbgcON			/*!<  Print debug messages via driver callback										*/
#if (SBGC_NEED_DEBUG)
	#define	SBGC_LOG_COMMAND_TIME	sbgcON
	#define	SBGC_LOG_COMMAND_NUMBER	sbgcON
	#define	SBGC_LOG_COMMAND_DIR	sbgcON
	#define	SBGC_LOG_COMMAND_NAME	sbgcON
	#define	SBGC_LOG_COMMAND_ID		sbgcON
	#define	SBGC_LOG_COMMAND_STATUS	sbgcON
	#define	SBGC_LOG_COMMAND_PARAM	sbgcON
	#define	SBGC_LOG_COMMAND_DATA	sbgcOFF
	#define	SBGC_CUSTOM_SPRINTF		sbgcOFF
#endif

#define		SBGC_NEED_ASSERTS		sbgcOFF
#define		SBGC_NEED_CONFIRM_CMD	sbgcON			/*!<  Handle CMD_CONFIRM (used by our shim)										*/
#define		SBGC_NEED_REF_INFO		sbgcOFF

#define		SBGC_DEFAULT_TIMEOUT	100				/*!<  ms. Default timeout for serial commands										*/
#define		SBGC_STARTUP_DELAY		500				/*!<  ms. Delay after connect before talking										*/

#define		ROLL					0
#define		PITCH					1
#define		YAW						2

#if (SBGC_ADJVAR_MODULE)
	#define	SBGC_ADJ_VARS_REF_INFO	sbgcON
	#define	SBGC_ADJ_VARS_NAMES		sbgcON
	#define	SBGC_ADJ_VARS_ADD_FLAG	sbgcOFF
#endif

#if (SBGC_ADJ_VARS_ADD_FLAG)
	typedef struct
	{
		unsigned char	parameter1;
	}	sbgcAdjVarCustFld_t;
#endif

#if (SBGC_NEED_DEBUG && SBGC_NEED_CONFIRM_CMD)
	#define	SBGC_DETAILED_CONFIRM	sbgcOFF
#endif


/* ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
 *						   OS Support Configurations
 */
#define		SBGC_USE_AZURE_RTOS		sbgcOFF
#define		SBGC_USE_FREE_RTOS		sbgcOFF
#define		SBGC_USE_PTHREAD_OS		sbgcOFF

#if (SBGC_USE_AZURE_RTOS || SBGC_USE_FREE_RTOS || SBGC_USE_PTHREAD_OS)
	#if ((SBGC_USE_PTHREAD_OS == sbgcOFF) || SBGC_USES_DOXYGEN)
		#define	SBGC_THREAD_STACK_SIZE		256
		#define	SBGC_THREAD_PRIOR			3
		#define	SBGC_THREAD_QUIET_PRIOR		1
	#endif

	#define	SBGC_NEED_AUTO_PING		sbgcOFF

	#if (SBGC_NEED_AUTO_PING)
		#define	SBGC_AUTO_PING_PERIOD		1000

		#if (SBGC_NEED_DEBUG)
			#define	SBGC_LOG_AUTO_PING		sbgcOFF
		#endif

		#if (SBGC_NEED_CALLBACKS)
			#define	SBGC_AUTO_PING_CALLBACK	sbgcOFF
		#endif
	#endif
#endif


/* ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
 *							   Driver Configurations
 */
#define		SBGC_USE_ARDUINO_DRIVER	sbgcOFF
#define		SBGC_USE_ESPIDF_DRIVER	sbgcOFF
#define		SBGC_USE_LINUX_DRIVER	sbgcON			/*!<  Using a Linux machine as a master device										*/
#define		SBGC_USE_STM32_DRIVER	sbgcOFF


#if (SBGC_USE_ARDUINO_DRIVER && (SBGC_SEVERAL_DEVICES == sbgcOFF))
	/* Arduino driver config (unused for us) */
	#define	SBGC_SERIAL_PORT		Serial1
	#if (SBGC_NEED_DEBUG)
		#define	SBGC_DEBUG_SERIAL_PORT	Serial
	#endif
	#define	SBGC_SERIAL_SPEED		115200
	#if (SBGC_NEED_DEBUG)
		#define	SBGC_DEBUG_SERIAL_SPEED	115200
	#endif
	#define	SBGC_SERIAL_TX_PIN		21
	#define	SBGC_SERIAL_RX_PIN		20
#endif /* SBGC_USE_ARDUINO_DRIVER */


#if (SBGC_USE_ESPIDF_DRIVER && (SBGC_SEVERAL_DEVICES == sbgcOFF))
	/* ESP-IDF driver config (unused for us) */
	#define	SBGC_DRV_TX_BUFFER_SIZE	1
	#define	SBGC_DRV_RX_BUFFER_SIZE	1

	#define	SBGC_SERIAL_PORT		UART_NUM_1
	#if (SBGC_NEED_DEBUG)
		#define	SBGC_DEBUG_SERIAL_PORT	UART_NUM_0
	#endif
	#define	SBGC_SERIAL_SPEED		115200
	#if (SBGC_NEED_DEBUG)
		#define	SBGC_DEBUG_SERIAL_SPEED	115200
	#endif
	#define	SBGC_SERIAL_TX_PIN		21
	#define	SBGC_SERIAL_RX_PIN		20
#endif /* SBGC_USE_ESPIDF_DRIVER */


#if (SBGC_USE_LINUX_DRIVER && (SBGC_SEVERAL_DEVICES == sbgcOFF))
	/**	@addtogroup	Linux_Driver
	 *	@{
	 */
	/*	Attention!
		The serial port must have extended rights:
		sudo chmod a+rwx /dev/ttyUSB0
	 */
	#define SBGC_SERIAL_PORT		"/dev/ttyUSB0"	/*!<  Path to a connected SBGC32 device												*/
	#define SBGC_SERIAL_SPEED		B115200			/*!<  SBGCs COM-port serial speed													*/
	/**	@}
	 */
#endif /* SBGC_USE_LINUX_DRIVER */


#if (SBGC_USE_STM32_DRIVER)
	/* STM32 driver config (unused for us) */
	#define	SBGC_SERIAL_SPEED		115200
	#if (!(SBGC_USE_FREE_RTOS || SBGC_USE_AZURE_RTOS))
		#define	SBGC_DRV_HAL_TIMER	sbgcOFF
		#define	SBGC_DRV_LL_TIMER	sbgcOFF
	#endif

	#define	SBGC_DRV_HAL_NVIC_UART	sbgcOFF
	#define	SBGC_DRV_HAL_DMA_UART	sbgcOFF
	#define	SBGC_DRV_LL_NVIC_UART	sbgcOFF
	#define	SBGC_DRV_LL_DMA_UART	sbgcOFF

	#define	SBGC_DRV_TX_BUFFER_SIZE	1
	#define	SBGC_DRV_RX_BUFFER_SIZE	1

	#if (SBGC_NEED_DEBUG)
		#define	SBGC_DRV_USE_UART_DEBUG	sbgcOFF
	#endif

	#define	SBGC_DRV_CONFIGURED		sbgcOFF
#endif /* SBGC_USE_STM32_DRIVER */

/**	@}
 */

/*  = = = = = = = = = = = = = = = = = = = = = = = */
#ifdef 		__cplusplus
			}
#endif

#endif		/* SERIALAPI_CONFIG_H_ */
