// simplebgc_shim.c
// Minimal shim wrapping BaseCam SerialAPI for use from Python via ctypes.
//
// This version:
//   - Initializes ROLL, PITCH, YAW in ANGLE mode
//   - Exposes:
//        int  bgc_init(void);
//        int  bgc_set_motors(int on);
//        int  bgc_control_angles(float roll_deg, float pitch_deg, float yaw_deg);
//        void bgc_deinit(void);

#include <string.h>
#include "sbgc32.h"

static sbgcGeneral_t       gSBGC;
static sbgcControl_t       gCtrl;
static sbgcControlConfig_t gCtrlCfg;
static sbgcConfirm_t       gConfirm;

int bgc_init(void)
{
    sbgcCommandStatus_t st;

    // Initialize main SerialAPI context
    st = SBGC32_Init(&gSBGC);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // -------- Control config (filters, etc.) --------
    memset(&gCtrlCfg, 0, sizeof(gCtrlCfg));

    // Enable some low-pass filtering for all three axes
    gCtrlCfg.AxisCCtrl[ROLL].angleLPF  = 2;
    gCtrlCfg.AxisCCtrl[PITCH].angleLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].angleLPF   = 2;

    gCtrlCfg.AxisCCtrl[ROLL].speedLPF  = 2;
    gCtrlCfg.AxisCCtrl[PITCH].speedLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].speedLPF   = 2;

    // No CMD_CONFIRM responses required
    gCtrlCfg.flags = CtrlCONFIG_FLAG_NO_CONFIRM;

    st = SBGC32_ControlConfig(&gSBGC, &gCtrlCfg, &gConfirm);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // -------- Control structure (modes, speeds, initial angles) --------
    memset(&gCtrl, 0, sizeof(gCtrl));

    // Use ANGLE mode on all three axes with "precise target" flag
    gCtrl.mode[ROLL]  = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    // Start from 0 deg on all axes
    gCtrl.AxisC[ROLL].angle  = sbgcDegreeToAngle(0.0f);
    gCtrl.AxisC[PITCH].angle = sbgcDegreeToAngle(0.0f);
    gCtrl.AxisC[YAW].angle   = sbgcDegreeToAngle(0.0f);

    // Modest speeds; tune as needed
    gCtrl.AxisC[ROLL].speed  = sbgcSpeedToValue(25.0f);
    gCtrl.AxisC[PITCH].speed = sbgcSpeedToValue(25.0f);
    gCtrl.AxisC[YAW].speed   = sbgcSpeedToValue(50.0f);

    // Turn motors ON once at init
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

// Updated: now takes roll, pitch, yaw *in degrees*.
// Python side should call bgc_control_angles(roll_deg, pitch_deg, yaw_deg).
int bgc_control_angles(float roll_deg, float pitch_deg, float yaw_deg)
{
    // Convert degrees → internal "angle units"
    gCtrl.AxisC[ROLL].angle  = sbgcDegreeToAngle(roll_deg);
    gCtrl.AxisC[PITCH].angle = sbgcDegreeToAngle(pitch_deg);
    gCtrl.AxisC[YAW].angle   = sbgcDegreeToAngle(yaw_deg);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}

void bgc_deinit(void)
{
    SBGC32_Deinit(&gSBGC);
}
