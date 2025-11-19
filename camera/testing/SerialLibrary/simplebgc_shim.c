// simplebgc_shim.c
#include "sbgc32.h"
#include <string.h>

/*
 * Thin C shim to expose a small, stable API to Python via ctypes.
 * Internally uses the official SimpleBGC SerialAPI library.
 *
 * Functions exported:
 *   int bgc_init(void);
 *   int bgc_setup_control_config(void);
 *   int bgc_set_motors(int on);
 *   int bgc_play_beep(void);
 *   int bgc_control_angles(double roll_deg, double pitch_deg, double yaw_deg);
 */

static sbgcGeneral_t         gSBGC;
static sbgcControl_t         gCtrl;
static sbgcControlConfig_t   gCtrlCfg;
static sbgcBeeperSettings_t  gBeep;

/* ------------------------------------------------------------------ */
/*                          Initialization                            */
/* ------------------------------------------------------------------ */

int bgc_init(void)
{
    sbgcCommandStatus_t st = SBGC32_Init(&gSBGC);

    if (st != sbgcCOMMAND_OK)
        return (int)st;

    return 0;
}

/*
 * Configure CMD_CONTROL in ANGLE mode for ROLL / PITCH / YAW.
 * Very similar in spirit to the official DemoLaunch example.
 */
int bgc_setup_control_config(void)
{
    memset(&gCtrlCfg, 0, sizeof(gCtrlCfg));
    memset(&gCtrl,    0, sizeof(gCtrl));

    /* Basic LPFs on angles and speeds */
    gCtrlCfg.AxisCCtrl[ROLL].angleLPF  = 2;
    gCtrlCfg.AxisCCtrl[PITCH].angleLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].angleLPF   = 2;

    gCtrlCfg.AxisCCtrl[ROLL].speedLPF  = 2;
    gCtrlCfg.AxisCCtrl[PITCH].speedLPF = 2;
    gCtrlCfg.AxisCCtrl[YAW].speedLPF   = 2;

    /* Don’t require CMD_CONFIRM for control config */
    gCtrlCfg.flags = CtrlCONFIG_FLAG_NO_CONFIRM;

    /* Commit the control config to the board */
    SBGC32_ControlConfig(&gSBGC, &gCtrlCfg, SBGC_NO_CONFIRM);

    /* Set control mode: ANGLE + PRECISE target for all axes */
    gCtrl.mode[ROLL]  = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    gCtrl.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    /* Start from zero angles, with reasonable speeds */
    gCtrl.AxisC[ROLL].angle  = sbgcAngleToDegree(0.0f);
    gCtrl.AxisC[PITCH].angle = sbgcAngleToDegree(0.0f);
    gCtrl.AxisC[YAW].angle   = sbgcAngleToDegree(0.0f);

    gCtrl.AxisC[ROLL].speed  = sbgcSpeedToValue(25.0f);
    gCtrl.AxisC[PITCH].speed = sbgcSpeedToValue(25.0f);
    gCtrl.AxisC[YAW].speed   = sbgcSpeedToValue(50.0f);

    return 0;
}

/* ------------------------------------------------------------------ */
/*                        Motors & Beeper helpers                      */
/* ------------------------------------------------------------------ */

/* on = 1 -> motors ON; on = 0 -> motors OFF */
int bgc_set_motors(int on)
{
    if (on)
    {
        /* Turn motors ON, no confirmation needed */
        return (int)SBGC32_SetMotorsON(&gSBGC, SBGC_NO_CONFIRM);
    }
    else
    {
        /* Turn motors OFF in normal mode, no confirmation needed */
        return (int)SBGC32_SetMotorsOFF(&gSBGC, MOTOR_MODE_NORMAL, SBGC_NO_CONFIRM);
    }
}

/* Simple “completion” beep using the built-in beeper feature */
int bgc_play_beep(void)
{
    memset(&gBeep, 0, sizeof(gBeep));
    gBeep.mode = BEEP_MODE_COMPLETE;

    return (int)SBGC32_PlayBeeper(&gSBGC, &gBeep, SBGC_NO_CONFIRM);
}

/* ------------------------------------------------------------------ */
/*                        Main control function                        */
/* ------------------------------------------------------------------ */

/*
 * roll_deg, pitch_deg, yaw_deg are ABSOLUTE target angles in degrees.
 * You can maintain the commanded angle state on the Python side and
 * call this whenever you want to step to a new absolute pose.
 */
int bgc_control_angles(double roll_deg, double pitch_deg, double yaw_deg)
{
    gCtrl.AxisC[ROLL].angle  = sbgcAngleToDegree((float)roll_deg);
    gCtrl.AxisC[PITCH].angle = sbgcAngleToDegree((float)pitch_deg);
    gCtrl.AxisC[YAW].angle   = sbgcAngleToDegree((float)yaw_deg);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}
