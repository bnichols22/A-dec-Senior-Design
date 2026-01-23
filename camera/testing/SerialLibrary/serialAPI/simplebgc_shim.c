// simplebgc_shim.c
// Minimal shim to read CURRENT camera angles (roll, pitch, yaw)
// Uses the SAME method as the official SimpleBGC example

#include <string.h>
#include <stdint.h>
#include "sbgc32.h"

static sbgcGeneral_t       gSBGC;
static sbgcControl_t       gCtrl;
static sbgcControlConfig_t gCtrlCfg;
static sbgcConfirm_t       gConfirm;

/* -------------------------------------------------- */
/* Init */
/* -------------------------------------------------- */
int bgc_init(void)
{
    sbgcCommandStatus_t st;

    st = SBGC32_Init(&gSBGC);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    memset(&gCtrlCfg, 0, sizeof(gCtrlCfg));

    gCtrlCfg.AxisCCtrl[PITCH].angleLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].angleLPF   = 2;
    gCtrlCfg.AxisCCtrl[PITCH].speedLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].speedLPF   = 2;

    gCtrlCfg.flags = CtrlCONFIG_FLAG_NO_CONFIRM;

    st = SBGC32_ControlConfig(&gSBGC, &gCtrlCfg, &gConfirm);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    memset(&gCtrl, 0, sizeof(gCtrl));
    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    gCtrl.AxisC[PITCH].speed = sbgcSpeedToValue(25.0f);
    gCtrl.AxisC[YAW].speed   = sbgcSpeedToValue(50.0f);

    /* Turn motors ON (important!) */
    st = SBGC32_SetMotorsON(&gSBGC, &gConfirm);
    return (int)st;
}

/* -------------------------------------------------- */
/* ONE-SHOT ANGLE POLL (THIS IS THE IMPORTANT PART) */
/* -------------------------------------------------- */
int bgc_get_angles(float *roll_deg, float *pitch_deg, float *yaw_deg)
{
    sbgcRealTimeData_t rtd;
    sbgcCommandStatus_t st;

    if (!roll_deg || !pitch_deg || !yaw_deg)
        return (int)sbgcCOMMAND_RX_ERROR;

    memset(&rtd, 0, sizeof(rtd));

    /* EXACTLY like the official example */
    st = SBGC32_ReadRealTimeData4(&gSBGC, &rtd);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    /* Convert controller units → degrees */
    *roll_deg  = sbgcAngleToDegree(rtd.frameCamAngle[ROLL]);
    *pitch_deg = sbgcAngleToDegree(rtd.frameCamAngle[PITCH]);
    *yaw_deg   = sbgcAngleToDegree(rtd.frameCamAngle[YAW]);

    return 0;
}

/* -------------------------------------------------- */
/* Command angles (unchanged) */
/* -------------------------------------------------- */
int bgc_control_angles(float roll_deg, float pitch_deg, float yaw_deg)
{
    gCtrl.AxisC[ROLL].angle  = sbgcDegreeToAngle(roll_deg);
    gCtrl.AxisC[PITCH].angle = sbgcDegreeToAngle(pitch_deg);
    gCtrl.AxisC[YAW].angle   = sbgcDegreeToAngle(yaw_deg);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}

void bgc_deinit(void)
{
    SBGC32_Deinit(&gSBGC);
}
