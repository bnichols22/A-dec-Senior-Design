// simplebgc_shim.c
// Basic working shim + angle readback (pitch,yaw order preserved)

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

    gCtrl.AxisC[PITCH].angle = 0;
    gCtrl.AxisC[YAW].angle   = 0;

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

int bgc_control_angles(float pitch_deg, float yaw_deg)
{
    // IMPORTANT FIX:
    // You were previously using sbgcAngleToDegree() here (backwards).
    // Command requires: degrees -> internal angle units.
    gCtrl.AxisC[PITCH].angle = sbgcDegreeToAngle(pitch_deg);
    gCtrl.AxisC[YAW].angle   = sbgcDegreeToAngle(yaw_deg);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}

/**
 * Read current gimbal angles from the board.
 * Output order matches the rest of the shim: (pitch, yaw).
 *
 * @param pitch_deg_out pointer to store pitch in degrees (required)
 * @param yaw_deg_out   pointer to store yaw in degrees (required)
 * @return 0 on success, otherwise SBGC status code
 */
int bgc_get_angles(float *pitch_deg_out, float *yaw_deg_out)
{
    sbgcRealTimeData_t rtd;
    sbgcCommandStatus_t st;

    if (!pitch_deg_out || !yaw_deg_out)
        return (int)sbgcCOMMAND_RX_ERROR;

    memset(&rtd, 0, sizeof(rtd));

    // One-shot read of realtime data (includes frameCamAngle[])
    st = SBGC32_ReadRealTimeData4(&gSBGC, &rtd);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // Convert internal angle units -> degrees
    *pitch_deg_out = sbgcAngleToDegree(rtd.frameCamAngle[PITCH]);
    *yaw_deg_out   = sbgcAngleToDegree(rtd.frameCamAngle[YAW]);

    return 0;
}

void bgc_deinit(void)
{
    SBGC32_Deinit(&gSBGC);
}
