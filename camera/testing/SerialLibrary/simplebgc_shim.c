// simplebgc_shim.c
// Thin C shim around BaseCam SerialAPI for use with Python (ctypes)

#include "sbgc32.h"
#include <string.h>   // memset
#include <stdio.h>    // printf (for debug)

// Global SBGC object and some commonly used structs
static sbgcGeneral_t           gSBGC;
static sbgcControl_t           gCtrl;
static sbgcControlConfig_t     gCtrlCfg;
static sbgcBeeperSettings_t    gBeep;

/* ------------- Utility: basic init / deinit ------------- */

int bgc_init(void)
{
    sbgcCommandStatus_t status;

    // Clear structures
    memset(&gSBGC,   0, sizeof(gSBGC));
    memset(&gCtrl,   0, sizeof(gCtrl));
    memset(&gCtrlCfg,0, sizeof(gCtrlCfg));
    memset(&gBeep,   0, sizeof(gBeep));

    // Init driver + library (see demo main.c pattern)
    status = SBGC32_Init(&gSBGC);
    if (status != sbgcCOMMAND_OK)
    {
        printf("SBGC32_Init failed: %d\n", (int)status);
        return (int)status;
    }

    // Configure control mode: all axes in ANGLE mode,
    // use board profile speeds, no special flags.
    gCtrlCfg.mode[ROLL]  = SBGC_CONTROL_MODE_ANGLE;
    gCtrlCfg.mode[PITCH] = SBGC_CONTROL_MODE_ANGLE;
    gCtrlCfg.mode[YAW]   = SBGC_CONTROL_MODE_ANGLE;

    gCtrlCfg.speeds[ROLL]  = 0;
    gCtrlCfg.speeds[PITCH] = 0;
    gCtrlCfg.speeds[YAW]   = 0;

    status = SBGC32_ControlConfig(&gSBGC, &gCtrlCfg, NULL);
    if (status != sbgcCOMMAND_OK)
    {
        printf("SBGC32_ControlConfig failed: %d\n", (int)status);
        return (int)status;
    }

    // Turn motors ON
    status = SBGC32_SetMotorsON(&gSBGC, NULL);
    if (status != sbgcCOMMAND_OK)
    {
        printf("SBGC32_SetMotorsON failed: %d\n", (int)status);
        return (int)status;
    }

    return 0;   // success
}

void bgc_deinit(void)
{
    // Turn motors OFF (normal mode) and let the library deinit
    SBGC32_SetMotorsOFF(&gSBGC, SBGC_MM_NORMAL, NULL);
    SBGC32_Deinit(&gSBGC);
}

/* ------------- Buzzer helper ------------- */

int bgc_beep_once(void)
{
    sbgcCommandStatus_t status;

    memset(&gBeep, 0, sizeof(gBeep));
    gBeep.beepNum   = 1;
    gBeep.beepCnt   = 1;
    gBeep.beepTime  = 200;   // ms
    gBeep.silTime   = 100;   // ms
    gBeep.mode      = SBGC_BEEP_MODE_FIXED;

    status = SBGC32_PlayBeeper(&gSBGC, &gBeep, NULL);
    return (int)status;
}

/* ------------- Core: angle control ------------- */
/* Angles in degrees, relative commands */

int bgc_control_angles(float yaw_deg, float pitch_deg, float roll_deg)
{
    sbgcCommandStatus_t status;

    memset(&gCtrl, 0, sizeof(gCtrl));

    gCtrl.mode[ROLL]  = SBGC_CONTROL_MODE_ANGLE;
    gCtrl.mode[PITCH] = SBGC_CONTROL_MODE_ANGLE;
    gCtrl.mode[YAW]   = SBGC_CONTROL_MODE_ANGLE;

    gCtrl.angle[ROLL]  = (sbgcAngle_t)roll_deg;
    gCtrl.angle[PITCH] = (sbgcAngle_t)pitch_deg;
    gCtrl.angle[YAW]   = (sbgcAngle_t)yaw_deg;

    // speeds = 0 => use profile
    gCtrl.speed[ROLL]  = 0;
    gCtrl.speed[PITCH] = 0;
    gCtrl.speed[YAW]   = 0;

    status = SBGC32_Control(&gSBGC, &gCtrl, NULL);
    return (int)status;
}

/* ------------- Optional: get current frame angles ------------- */

int bgc_get_angles(float *yaw_deg, float *pitch_deg, float *roll_deg)
{
    sbgcRealTimeData_t rtData;
    sbgcCommandStatus_t status;

    memset(&rtData, 0, sizeof(rtData));
    status = SBGC32_ReadRealTimeData4(&gSBGC, &rtData, NULL);
    if (status != sbgcCOMMAND_OK)
        return (int)status;

    if (roll_deg)  *roll_deg  = (float)rtData.frameAngle[ROLL];
    if (pitch_deg) *pitch_deg = (float)rtData.frameAngle[PITCH];
    if (yaw_deg)   *yaw_deg   = (float)rtData.frameAngle[YAW];

    return 0;
}
