// simplebgc_shim.c
// Minimal shim around SimpleBGC SerialAPI for use from Python (ctypes).
// Now supports ROLL + PITCH + YAW absolute angle control.

#include <string.h>
#include <stdint.h>
#include "sbgc32.h"

static sbgcGeneral_t       gSBGC;
static sbgcControl_t       gCtrl;
static sbgcControlConfig_t gCtrlCfg;
static sbgcConfirm_t       gConfirm;

/*
 * Initialize SerialAPI, basic control config, and motors ON.
 * Returns 0 on success, or a sbgcCommandStatus_t code on error.
 */
int bgc_init(void)
{
    sbgcCommandStatus_t st;

    // Initialize the library + Linux driver (uses SBGC_SERIAL_PORT from serialAPI_Config.h)
    st = SBGC32_Init(&gSBGC);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // ----- ControlConfig -----
    memset(&gCtrlCfg, 0, sizeof(gCtrlCfg));

    // Enable LPF on all three axes (ROLL, PITCH, YAW)
    gCtrlCfg.AxisCCtrl[ROLL].angleLPF  = 2;
    gCtrlCfg.AxisCCtrl[PITCH].angleLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].angleLPF   = 2;

    gCtrlCfg.AxisCCtrl[ROLL].speedLPF  = 2;
    gCtrlCfg.AxisCCtrl[PITCH].speedLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].speedLPF   = 2;

    // Don't require CMD_CONFIRM back for each control command
    gCtrlCfg.flags = CtrlCONFIG_FLAG_NO_CONFIRM;

    st = SBGC32_ControlConfig(&gSBGC, &gCtrlCfg, &gConfirm);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // ----- Control object -----
    memset(&gCtrl, 0, sizeof(gCtrl));

    // Angle-control, “target precise” on ROLL, PITCH, YAW
    gCtrl.mode[ROLL]  = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    // Start at 0 deg on all axes
    gCtrl.AxisC[ROLL].angle  = sbgcAngleToDegree(0);
    gCtrl.AxisC[PITCH].angle = sbgcAngleToDegree(0);
    gCtrl.AxisC[YAW].angle   = sbgcAngleToDegree(0);

    // Reasonable speeds (deg/s → internal units)
    gCtrl.AxisC[ROLL].speed  = sbgcSpeedToValue(25.0f);
    gCtrl.AxisC[PITCH].speed = sbgcSpeedToValue(25.0f);
    gCtrl.AxisC[YAW].speed   = sbgcSpeedToValue(50.0f);

    // Turn motors on
    st = SBGC32_SetMotorsON(&gSBGC, &gConfirm);
    return (int)st;
}

/*
 * Turn motors on (on=1) or off (on=0).
 */
int bgc_set_motors(int on)
{
    sbgcCommandStatus_t st;

    if (on)
        st = SBGC32_SetMotorsON(&gSBGC, &gConfirm);
    else
        st = SBGC32_SetMotorsOFF(&gSBGC, 0, &gConfirm);

    return (int)st;
}

/*
 * Set absolute angles (in degrees) for all three axes.
 * roll_deg, pitch_deg, yaw_deg are in mechanical degrees.
 */
int bgc_control_angles(float roll_deg, float pitch_deg, float yaw_deg)
{
    // SimpleBGC expects 'angle' field in its internal "degree units"
    // (1 LSB ≈ 0.02197 deg). sbgcAngleToDegree() converts from degrees → internal.
    gCtrl.AxisC[ROLL].angle  = sbgcAngleToDegree((int16_t)roll_deg);
    gCtrl.AxisC[PITCH].angle = sbgcAngleToDegree((int16_t)pitch_deg);
    gCtrl.AxisC[YAW].angle   = sbgcAngleToDegree((int16_t)yaw_deg);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}

/*
 * Deinit and close serial port.
 */
void bgc_deinit(void)
{
    SBGC32_Deinit(&gSBGC);
}
