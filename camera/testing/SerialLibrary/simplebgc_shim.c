// simplebgc_shim.c
// Minimal C shim around SerialAPI to control ROLL, PITCH, YAW
// via absolute angles in degrees.

#include <string.h>
#include "sbgc32.h"

/* Global objects (mirroring demo main.c style) */
static sbgcGeneral_t       SBGC32_Device;
static sbgcControl_t       Control;
static sbgcControlConfig_t ControlConfig;
static sbgcConfirm_t       Confirm;

/* ------------------------------------------------------------------
 * bgc_init
 *  - Initializes SerialAPI
 *  - Sets up ControlConfig and Control for ROLL, PITCH, YAW
 *  - Turns motors ON
 *  Returns 0 on success, non-zero sbgcCommandStatus_t on error.
 * ------------------------------------------------------------------ */
int bgc_init(void)
{
    sbgcCommandStatus_t st;

    /* Init library and driver */
    st = SBGC32_Init(&SBGC32_Device);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    /* Clear control config and control structures */
    memset(&ControlConfig, 0, sizeof(ControlConfig));
    memset(&Control,       0, sizeof(Control));

    /* Low-pass filters, similar to demo but now for all 3 axes */
    ControlConfig.AxisCCtrl[ROLL].angleLPF  = 2;
    ControlConfig.AxisCCtrl[PITCH].angleLPF = 2;
    ControlConfig.AxisCCtrl[YAW].angleLPF   = 2;

    ControlConfig.AxisCCtrl[ROLL].speedLPF  = 2;
    ControlConfig.AxisCCtrl[PITCH].speedLPF = 2;
    ControlConfig.AxisCCtrl[YAW].speedLPF   = 2;

    /* No confirmation on each control command */
    ControlConfig.flags = CtrlCONFIG_FLAG_NO_CONFIRM;

    /* Tell board about this configuration */
    st = SBGC32_ControlConfig(&SBGC32_Device, &ControlConfig, &Confirm);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    /* Enable ANGLE mode on ALL THREE axes, with precise targeting */
    Control.mode[ROLL]  = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    Control.mode[PITCH] = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;
    Control.mode[YAW]   = CtrlMODE_ANGLE | CtrlFLAG_TARGET_PRECISE;

    /* Start at 0° on all axes */
    Control.AxisC[ROLL].angle  = sbgcAngleToDegree(0.0f);
    Control.AxisC[PITCH].angle = sbgcAngleToDegree(0.0f);
    Control.AxisC[YAW].angle   = sbgcAngleToDegree(0.0f);

    /* Reasonable speeds (deg/s → internal units) */
    Control.AxisC[ROLL].speed  = sbgcSpeedToValue(25.0f);
    Control.AxisC[PITCH].speed = sbgcSpeedToValue(25.0f);
    Control.AxisC[YAW].speed   = sbgcSpeedToValue(50.0f);

    /* Turn motors ON */
    st = SBGC32_SetMotorsON(&SBGC32_Device, &Confirm);
    return (int)st;
}

/* ------------------------------------------------------------------
 * bgc_set_motors
 *  on = 1 → motors ON
 *  on = 0 → motors OFF
 * ------------------------------------------------------------------ */
int bgc_set_motors(int on)
{
    sbgcCommandStatus_t st;

    if (on)
        st = SBGC32_SetMotorsON(&SBGC32_Device, &Confirm);
    else
        st = SBGC32_SetMotorsOFF(&SBGC32_Device, 0, &Confirm);

    return (int)st;
}

/* ------------------------------------------------------------------
 * bgc_control_angles
 *  roll_deg, pitch_deg, yaw_deg are ABSOLUTE target angles in degrees.
 *  We convert to internal units exactly like the demo does:
 *      Control.AxisC[*].angle = sbgcAngleToDegree(deg);
 * ------------------------------------------------------------------ */
int bgc_control_angles(float roll_deg, float pitch_deg, float yaw_deg)
{
    /* Convert degrees → internal angle units, as in DemoControl() */
    Control.AxisC[ROLL].angle  = sbgcAngleToDegree(roll_deg);
    Control.AxisC[PITCH].angle = sbgcAngleToDegree(pitch_deg);
    Control.AxisC[YAW].angle   = sbgcAngleToDegree(yaw_deg);

    return (int)SBGC32_Control(&SBGC32_Device, &Control);
}

/* ------------------------------------------------------------------
 * bgc_deinit
 * ------------------------------------------------------------------ */
void bgc_deinit(void)
{
    SBGC32_Deinit(&SBGC32_Device);
}
