#include <string.h>
#include "sbgc32.h"

/*
 * Very small C shim we can call from Python.
 * It uses the same logic as the DemoLaunch example:
 * - init the device
 * - set up ControlConfig
 * - enable motors
 * - send simple angle commands on Pitch/Yaw
 */

static sbgcGeneral_t       gSBGC;
static sbgcControl_t       gCtrl;
static sbgcControlConfig_t gCtrlCfg;
static sbgcConfirm_t       gConfirm;   // we’ll mostly ignore its contents

/* 0 = OK, non-zero = sbgcCommandStatus_t error code */
int bgc_init(void)
{
    sbgcCommandStatus_t st;

    // Initialize the library (uses Linux driver & /dev/ttyUSB0 from serialAPI_Config.h)
    st = SBGC32_Init(&gSBGC);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // ---- Control configuration (mirrors the demo)
    memset(&gCtrlCfg, 0, sizeof(gCtrlCfg));

    gCtrlCfg.AxisCCtrl[PITCH].angleLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].angleLPF   = 2;

    gCtrlCfg.AxisCCtrl[PITCH].speedLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].speedLPF   = 2;

    gCtrlCfg.flags = CtrlCONFIG_FLAG_NO_CONFIRM;

    st = SBGC32_ControlConfig(&gSBGC, &gCtrlCfg, &gConfirm);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // ---- Base control struct
    memset(&gCtrl, 0, sizeof(gCtrl));

    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    gCtrl.AxisC[PITCH].angle = 0;
    gCtrl.AxisC[YAW].angle   = 0;

    // Speeds in “system units” (see docs, same as demo)
    gCtrl.AxisC[PITCH].speed = sbgcSpeedToValue(25.0f);
    gCtrl.AxisC[YAW].speed   = sbgcSpeedToValue(50.0f);

    // ---- Turn motors ON
    st = SBGC32_SetMotorsON(&gSBGC, &gConfirm);
    return (int)st;
}

/* Turn motors on/off: on = 1 -> ON, on = 0 -> OFF */
int bgc_set_motors(int on)
{
    sbgcCommandStatus_t st;

    if (on)
        st = SBGC32_SetMotorsON(&gSBGC, &gConfirm);
    else
        st = SBGC32_SetMotorsOFF(&gSBGC, 0, &gConfirm);  // 0 = default stop mode

    return (int)st;
}

/* Move gimbal to given angles (degrees) on PITCH & YAW */
int bgc_control_angles(float pitch_deg, float yaw_deg)
{
    // Library expects “system degree” units; helper macro converts from human degrees
    gCtrl.AxisC[PITCH].angle = sbgcAngleToDegree((int16_t)pitch_deg);
    gCtrl.AxisC[YAW].angle   = sbgcAngleToDegree((int16_t)yaw_deg);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}

void bgc_deinit(void)
{
    SBGC32_Deinit(&gSBGC);
}
