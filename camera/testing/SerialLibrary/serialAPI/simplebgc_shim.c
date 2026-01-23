// simplebgc_shim.c
// Basic working shim + ONE-SHOT encoder/motor angle poll via RT data custom stream
// Command order preserved: bgc_control_angles(pitch, yaw)
// Readback order preserved: bgc_get_angles(pitch, yaw, roll_optional)

#include <string.h>
#include <stdint.h>
#include <time.h>

#include "sbgc32.h"

static sbgcGeneral_t       gSBGC;
static sbgcControl_t       gCtrl;
static sbgcControlConfig_t gCtrlCfg;
static sbgcConfirm_t       gConfirm;

/* One-shot RT data custom packet for STATOR/ROTOR angles */
struct PACKED__ RTOnceAnglesPacket
{
    ui16 timestampMs;          // mandatory
    i16  statorRotorAngle[3];  // ROLL, PITCH, YAW (as provided by RTDCF_STATOR_ROTOR_ANGLE)
};

/* small helper: monotonic-ish milliseconds */
static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000ULL);
}

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
    // degrees -> internal angle units
    gCtrl.AxisC[PITCH].angle = sbgcDegreeToAngle(pitch_deg);
    gCtrl.AxisC[YAW].angle   = sbgcDegreeToAngle(yaw_deg);

    return (int)SBGC32_Control(&gSBGC, &gCtrl);
}

/**
 * ONE-SHOT poll of motor/encoder angles using the same approach as the demo:
 * - Start RT data custom stream with RTDCF_STATOR_ROTOR_ANGLE
 * - Read exactly one packet
 * - Convert internal -> degrees
 *
 * Output order preserved (pitch, yaw). Optional roll is last so old callers stay sane.
 *
 * @param pitch_deg_out pointer to store pitch in degrees (required)
 * @param yaw_deg_out   pointer to store yaw in degrees (required)
 * @param roll_deg_out  pointer to store roll in degrees (optional, can be NULL)
 * @return 0 on success, otherwise SBGC status code
 */
int bgc_get_angles(float *pitch_deg_out, float *yaw_deg_out, float *roll_deg_out)
{
    if (!pitch_deg_out || !yaw_deg_out)
        return (int)sbgcCOMMAND_RX_ERROR;

    sbgcCommandStatus_t st;

    sbgcDataStreamInterval_t ds;
    memset(&ds, 0, sizeof(ds));
    ds.cmdID = DSC_CMD_REALTIME_DATA_CUSTOM;
    ds.intervalMs = 1000;          // doesn't matter much for one-shot
    ds.syncToData = STD_SYNC_OFF;

    // Request the stator/rotor angles (motor encoder angles)
    ParserSBGC32_RTDC_FlagsToStream(&ds, RTDCF_STATOR_ROTOR_ANGLE);

    // Start stream
    memset(&gConfirm, 0, sizeof(gConfirm));
    st = SBGC32_StartDataStream(&gSBGC, &ds, &gConfirm);
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // Confirm is optional depending on flags/config, but we can still proceed.
    // We'll just try to read one packet.

    struct RTOnceAnglesPacket pkt;
    memset(&pkt, 0, sizeof(pkt));

    // Wait for one full packet to arrive (with a timeout)
    const uint64_t t_start = now_ms();
    const uint64_t timeout_ms = 1500;

    const int need_bytes = (int)(sizeof(pkt) + SBGC_SERVICE_BYTES_NUM);

    while (SerialAPI_GetBytesAvailable(&gSBGC) < need_bytes)
    {
        if ((now_ms() - t_start) > timeout_ms)
        {
            // Timed out waiting for stream data
            return (int)sbgcCOMMAND_RX_TIMEOUT;
        }
        sbgcDelay(2);
    }

    // Read exactly one packet
    st = SBGC32_ReadDataStream(&gSBGC, DSC_CMD_REALTIME_DATA_CUSTOM, &pkt, sizeof(pkt));
    if (st != sbgcCOMMAND_OK)
        return (int)st;

    // Convert internal angle units -> degrees
    const float roll_deg  = sbgcAngleToDegree(pkt.statorRotorAngle[ROLL]);
    const float pitch_deg = sbgcAngleToDegree(pkt.statorRotorAngle[PITCH]);
    const float yaw_deg   = sbgcAngleToDegree(pkt.statorRotorAngle[YAW]);

    *pitch_deg_out = pitch_deg;
    *yaw_deg_out   = yaw_deg;
    if (roll_deg_out)
        *roll_deg_out = roll_deg;

    return 0;
}

void bgc_deinit(void)
{
    SBGC32_Deinit(&gSBGC);
}
