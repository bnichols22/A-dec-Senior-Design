// simplebgc_shim.c
// Minimal shim around SimpleBGC SerialAPI for use from Python (ctypes).
// Supports absolute ANGLE control (existing) + SPEED control (new).
// SPEED control avoids the initial "jump to 0" bug because it never commands an absolute angle.

#include <string.h>
#include <stdint.h>
#include "sbgc32.h"

static sbgcGeneral_t       gSBGC;
static sbgcControl_t       gCtrl;
static sbgcControlConfig_t gCtrlCfg;
static sbgcConfirm_t       gConfirm;

int bgc_init(void)
{
    sbgcCommandStatus_t st;

    st = SBGC32_Init(&gSBGC);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // ----- ControlConfig -----
    memset(&gCtrlCfg, 0, sizeof(gCtrlCfg));

    gCtrlCfg.AxisCCtrl[ROLL].angleLPF  = 2;
    gCtrlCfg.AxisCCtrl[PITCH].angleLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].angleLPF   = 2;

    gCtrlCfg.AxisCCtrl[ROLL].speedLPF  = 2;
    gCtrlCfg.AxisCCtrl[PITCH].speedLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].speedLPF   = 2;

    gCtrlCfg.flags = CtrlCONFIG_FLAG_NO_CONFIRM;

    st = SBGC32_ControlConfig(&gSBGC, &gCtrlCfg, &gConfirm);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // ----- Control object -----
    memset(&gCtrl, 0, sizeof(gCtrl));

    // Default to ANGLE mode (your existing behavior).
    gCtrl.mode[ROLL]  = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    // Start targets at 0 (but NOTE: gimbal will not move until you call Control)
    gCtrl.AxisC[ROLL].angle  = sbgcAngleToDegree(0);
    gCtrl.AxisC[PITCH].angle = sbgcAngleToDegree(0);
    gCtrl.AxisC[YAW].angle   = sbgcAngleToDegree(0);

    gCtrl.AxisC[ROLL].speed  = sbgcSpeedToValue(25.0f);
    gCtrl.AxisC[PITCH].speed = sbgcSpeedToValue(25.0f);
    gCtrl.AxisC[YAW].speed   = sbgcSpeedToValue(50.0f);

    st = SBGC32_SetMotorsON(&gSBGC, &gConfirm);
    return (int)st;
}

int bgc_set_motors(int on)
{
    sbgcCommandStatus_t st;

    if (on)
        st = SBGC32_SetMotorsON(&gSBGC, &gConfirm);
    else
        st = SBGC32_SetMotorsOFF(&gSBGC, 0, &gConfirm);

    return (int)st;
}

// Existing absolute-angle control (kept exactly in spirit).
int bgc_control_angles(float roll_deg, float pitch_deg, float yaw_deg)
{
    // NOTE: This naming in the vendor lib is confusing, but your demo uses sbgcAngleToDegree(x_degrees)
    // for degree->internal conversion, so we keep it consistent with that style.
    gCtrl.mode[ROLL]  = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    gCtrl.AxisC[ROLL].angle  = sbgcAngleToDegree((int16_t)roll_deg);
    gCtrl.AxisC[PITCH].angle = sbgcAngleToDegree((int16_t)pitch_deg);
    gCtrl.AxisC[YAW].angle   = sbgcAngleToDegree((int16_t)yaw_deg);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}

/**
 * NEW: Speed control in deg/sec for all three axes.
 * This avoids the startup snap-to-zero bug because we never command an absolute angle.
 *
 * roll_dps / pitch_dps / yaw_dps are in degrees per second.
 */
int bgc_control_speeds(float roll_dps, float pitch_dps, float yaw_dps)
{
    gCtrl.mode[ROLL]  = CtrlMODE_SPEED;
    gCtrl.mode[PITCH] = CtrlMODE_SPEED;
    gCtrl.mode[YAW]   = CtrlMODE_SPEED;

    gCtrl.AxisC[ROLL].speed  = sbgcSpeedToValue(roll_dps);
    gCtrl.AxisC[PITCH].speed = sbgcSpeedToValue(pitch_dps);
    gCtrl.AxisC[YAW].speed   = sbgcSpeedToValue(yaw_dps);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}

void bgc_deinit(void)
{
    SBGC32_Deinit(&gSBGC);
}
