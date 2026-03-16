// simplebgc_shim.c
// Minimal shim around SimpleBGC SerialAPI for use from Python (ctypes).
// Supports absolute ANGLE control + SPEED control + reading current angles.

#include <string.h>
#include <stdint.h>
#include "sbgc32.h"

static sbgcGeneral_t       gSBGC;
static sbgcControl_t       gCtrl;
static sbgcControlConfig_t gCtrlCfg;
static sbgcConfirm_t       gConfirm;
static sbgcGetAngles_t     gAngles;

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

    gCtrl.mode[ROLL]  = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    gCtrl.AxisC[ROLL].angle  = sbgcDegreeToAngle(0);
    gCtrl.AxisC[PITCH].angle = sbgcDegreeToAngle(0);
    gCtrl.AxisC[YAW].angle   = sbgcDegreeToAngle(0);

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

// Absolute-angle control.
// FIXED: use sbgcDegreeToAngle() for degrees -> internal units.
int bgc_control_angles(float roll_deg, float pitch_deg, float yaw_deg)
{
    gCtrl.mode[ROLL]  = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    gCtrl.AxisC[ROLL].angle  = sbgcDegreeToAngle(roll_deg);
    gCtrl.AxisC[PITCH].angle = sbgcDegreeToAngle(pitch_deg);
    gCtrl.AxisC[YAW].angle   = sbgcDegreeToAngle(yaw_deg);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}

// Speed control in deg/sec for all three axes.
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

// Read current IMU angles (actual gimbal attitude) in degrees.
int bgc_get_angles(float *roll_deg, float *pitch_deg, float *yaw_deg)
{
    sbgcCommandStatus_t st;

    if (!roll_deg || !pitch_deg || !yaw_deg)
        return -1;

    memset(&gAngles, 0, sizeof(gAngles));

    st = SBGC32_GetAngles(&gSBGC, &gAngles);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    *roll_deg  = sbgcAngleToDegree(gAngles.AxisGA[ROLL].IMU_Angle);
    *pitch_deg = sbgcAngleToDegree(gAngles.AxisGA[PITCH].IMU_Angle);
    *yaw_deg   = sbgcAngleToDegree(gAngles.AxisGA[YAW].IMU_Angle);

    return 0;
}

// Read controller target angles in degrees.
int bgc_get_target_angles(float *roll_deg, float *pitch_deg, float *yaw_deg)
{
    sbgcCommandStatus_t st;

    if (!roll_deg || !pitch_deg || !yaw_deg)
        return -1;

    memset(&gAngles, 0, sizeof(gAngles));

    st = SBGC32_GetAngles(&gSBGC, &gAngles);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    *roll_deg  = sbgcAngleToDegree(gAngles.AxisGA[ROLL].targetAngle);
    *pitch_deg = sbgcAngleToDegree(gAngles.AxisGA[PITCH].targetAngle);
    *yaw_deg   = sbgcAngleToDegree(gAngles.AxisGA[YAW].targetAngle);

    return 0;
}

void bgc_deinit(void)
{
    SBGC32_Deinit(&gSBGC);
}
