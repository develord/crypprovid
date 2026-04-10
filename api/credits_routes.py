"""
Credits Routes
===============
Get balance + spend credits + AdMob SSV callback
"""
from fastapi import APIRouter, HTTPException, Depends, Request, status
from datetime import datetime
import logging

from auth import get_current_user
from database import get_credits, add_credits, spend_credits, get_last_earn_time
from models import CreditsResponse, SpendCreditsRequest, SpendCreditsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/credits", tags=["Credits"])

EARN_COOLDOWN_SECONDS = 10  # Min time between ad rewards


@router.get("", response_model=CreditsResponse, summary="Get credit balance")
async def get_balance(current_user: dict = Depends(get_current_user)):
    """Get current user's credit balance"""
    data = await get_credits(current_user["id"])
    return data


@router.get("/admob-callback", summary="AdMob SSV callback")
async def admob_ssv_callback(request: Request):
    """AdMob Server-Side Verification — Google calls this after a rewarded ad.
    No JWT required — called by Google servers, not the app."""
    params = dict(request.query_params)
    user_id = params.get("user_id")
    custom_data = params.get("custom_data")

    logger.info(f"AdMob SSV callback: user_id={user_id}, custom_data={custom_data}, params={params}")

    if not user_id:
        logger.warning("AdMob SSV: missing user_id")
        return {"status": "error", "message": "missing user_id"}

    try:
        uid = int(user_id)

        # Anti-spam cooldown
        last_earn = await get_last_earn_time(uid)
        if last_earn:
            last_dt = datetime.fromisoformat(last_earn)
            elapsed = (datetime.utcnow() - last_dt).total_seconds()
            if elapsed < EARN_COOLDOWN_SECONDS:
                logger.warning(f"AdMob SSV: cooldown for user {uid}, {elapsed:.0f}s < {EARN_COOLDOWN_SECONDS}s")
                return {"status": "error", "message": "cooldown"}

        new_balance = await add_credits(uid, 3, "earn_ad")
        logger.info(f"AdMob SSV: user {uid} earned 3 credits, balance: {new_balance}")
        return {"status": "ok", "balance": new_balance}
    except Exception as e:
        logger.error(f"AdMob SSV error: {e}")
        return {"status": "error"}


@router.post("/spend", response_model=SpendCreditsResponse, summary="Spend credits on prediction")
async def spend_on_prediction(request: SpendCreditsRequest, current_user: dict = Depends(get_current_user)):
    """Spend credits to view a crypto prediction"""
    user_id = current_user["id"]

    new_balance = await spend_credits(user_id, 3, request.crypto)
    if new_balance is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient credits. Watch an ad to earn more."
        )

    logger.info(f"User {user_id} spent 3 credits on {request.crypto}, balance: {new_balance}")

    return {"success": True, "balance": new_balance, "crypto": request.crypto}
