// simplebgc_shim.c
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

    memset(&gCtrl, 0, sizeof(gCtrl));

    gCtrl.mode[ROLL]  = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    gCtrl.AxisC[ROLL].angle  = sbgcDegreeToAngle(0.0f);
    gCtrl.AxisC[PITCH].angle = sbgcDegreeToAngle(0.0f);
    gCtrl.AxisC[YAW].angle   = sbgcDegreeToAngle(0.0f);

    gCtrl.AxisC[ROLL].speed  = sbgcSpeedToValue(30.0f);
    gCtrl.AxisC[PITCH].speed = sbgcSpeedToValue(30.0f);
    gCtrl.AxisC[YAW].speed   = sbgcSpeedToValue(60.0f);

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

// LEFT UNCHANGED because this is the version you said works.
int bgc_control_angles(float roll_deg, float pitch_deg, float yaw_deg)
{
    gCtrl.mode[ROLL]  = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    gCtrl.AxisC[ROLL].angle  = sbgcAngleToDegree((int16_t)roll_deg);
    gCtrl.AxisC[PITCH].angle = sbgcAngleToDegree((int16_t)pitch_deg);
    gCtrl.AxisC[YAW].angle   = sbgcAngleToDegree((int16_t)yaw_deg);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}

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

// ---------- DEGREE READBACK ----------

int bgc_get_angles_deg(float *roll_deg, float *pitch_deg, float *yaw_deg)
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

int bgc_get_target_angles_deg(float *roll_deg, float *pitch_deg, float *yaw_deg)
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

// ---------- RAW ANGLE-UNIT READBACK ----------
// These return the controller's internal angle units directly.

int bgc_get_angles_raw(float *roll_angle_units, float *pitch_angle_units, float *yaw_angle_units)
{
    sbgcCommandStatus_t st;

    if (!roll_angle_units || !pitch_angle_units || !yaw_angle_units)
        return -1;

    memset(&gAngles, 0, sizeof(gAngles));

    st = SBGC32_GetAngles(&gSBGC, &gAngles);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    *roll_angle_units  = (float)gAngles.AxisGA[ROLL].IMU_Angle;
    *pitch_angle_units = (float)gAngles.AxisGA[PITCH].IMU_Angle;
    *yaw_angle_units   = (float)gAngles.AxisGA[YAW].IMU_Angle;

    return 0;
}

int bgc_get_target_angles_raw(float *roll_angle_units, float *pitch_angle_units, float *yaw_angle_units)
{
    sbgcCommandStatus_t st;

    if (!roll_angle_units || !pitch_angle_units || !yaw_angle_units)
        return -1;

    memset(&gAngles, 0, sizeof(gAngles));

    st = SBGC32_GetAngles(&gSBGC, &gAngles);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    *roll_angle_units  = (float)gAngles.AxisGA[ROLL].targetAngle;
    *pitch_angle_units = (float)gAngles.AxisGA[PITCH].targetAngle;
    *yaw_angle_units   = (float)gAngles.AxisGA[YAW].targetAngle;

    return 0;
}

// ---------- HOLD HELPERS ----------

// Hold based on raw controller angle units.
// This is the safest match for your current working bgc_control_angles().
int bgc_hold_current_position_raw(void)
{
    float roll_raw, pitch_raw, yaw_raw;
    int rc;

    rc = bgc_get_angles_raw(&roll_raw, &pitch_raw, &yaw_raw);
    if (rc != 0)
        return rc;

    rc = bgc_control_angles(roll_raw, pitch_raw, yaw_raw);
    return rc;
}

// Optional: hold based on degrees, if you want to experiment later.
int bgc_hold_current_position_deg(void)
{
    float roll_deg, pitch_deg, yaw_deg;
    int rc;

    rc = bgc_get_angles_deg(&roll_deg, &pitch_deg, &yaw_deg);
    if (rc != 0)
        return rc;

    rc = bgc_control_angles(roll_deg, pitch_deg, yaw_deg);
    return rc;
}

void bgc_deinit(void)
{
    SBGC32_Deinit(&gSBGC);
}
