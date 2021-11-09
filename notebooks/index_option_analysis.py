'''Utilities for dealing with SPX option chains'''

__all__ = []

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from scipy.interpolate import pchip_interpolate

_gaussian = scipy.stats.norm()
_N_cdf = _gaussian.cdf
_N_pdf = _gaussian.pdf
_N_invcdf = _gaussian.ppf

_DAYS_PER_YEAR = 365.

def fwd_greeks_from(fwd, strike, tau, sigma, is_call, ref_tau=1./12.):
    '''Computes Black-Scholes prices and greeks assuming zero interest and dividend.
    
    Inputs are assumed to be np.array or pandas.Series.

    Returns a dictionary 

    See 
    https://en.wikipedia.org/wiki/Greeks_(finance)
    '''
    sqrt_tau = np.sqrt(tau)
    log_fwd_over_strike = np.log(fwd/strike)
    d1 = (log_fwd_over_strike + 0.5*sigma**2 * tau) / (sigma*sqrt_tau)
    d2 = d1 - sqrt_tau * sigma
    
    Nd1 = _N_cdf(d1)
    phid1 = _N_pdf(d1)
    Nd2 = _N_cdf(d2)
    Nmd1 = _N_cdf(-d1)
    Nmd2 = _N_cdf(-d2)
    
    call_price = fwd * Nd1 - strike * Nd2
    put_price = -fwd * Nmd1 + strike * Nmd2
    
    call_delta = Nd1
    put_delta = -Nmd1
    
    # Note: in this special case, theta is the same for call and put options
    theta = -0.5*fwd * (sigma/sqrt_tau) * phid1 
    gamma = (phid1) / (fwd * sigma * sqrt_tau)
    speed = -(gamma / fwd) * (1. + d1 / (sigma * sqrt_tau))

    vega = fwd * sqrt_tau * phid1
    wt_vega = np.sqrt(ref_tau / tau) * vega # todo: check correctness
    vanna = (vega/fwd)*(1-d1/(sigma*sqrt_tau))
    vomma = vega * d1 * d2 / sigma
    ultima = (-vega/sigma**2)*(d1*d2*(1-d1*d2)+d1**2+d2**2)
    
    result = dict(  call_price=call_price,
                    put_price=put_price,
                    call_delta=call_delta,
                    put_delta=put_delta,
                    theta=theta,
                    gamma=gamma,
                    speed=speed,
                    vega=vega,
                    vanna=vanna,
                    vomma=vomma,
                    ultima=ultima,
                    wt_vega=wt_vega,
                )
    
    # todo: fill nan with zero before multiplying by bool
    result['price'] = 1.*is_call * call_price + (1.-is_call)*put_price
    result['delta'] = 1.*is_call * call_delta + (1.-is_call)*put_delta
    result['abs_delta'] = np.abs(result['delta'])

    return result

def refine_iv(tgt, price, vega, vomma, ultima, order=3):

    '''One iteration of Newton-like method for implied vol calculation

    A higher order generalization of Newton's method is supported.
    See https://en.wikipedia.org/wiki/Householder%27s_method

    Params
    ------
    tgt : (np.array) observed option price (calibration target)
    price : (np.array) model-computed price (call or put) given sigma
    vega, vomma, ultima : (np.array) model-computed greeks given sigma
    order : (int) 1=Newton's method, 2-3 are higher order Householder methods

    Returns
    -------
    An array (compatible with tgt) that, when added to the current
    implied volatility, gives an improved estimate. That is,
    iv -> iv + update.

    Notes
    -----
    The paper by Li (2006) provides a useful domain for when this
    type of iteration can be expected to converge:
    |x| <= 0.5, 0 << v <= 1, and |x/v| <= 2,
    where x = log(F/K), F=exp((r-q)*tau)*spot, and v = sqrt(tau)*sigma.

    Generally, starting with a sigma near the upper end of Li's domain
    gives good convergence rates.
    '''
    x = tgt - price
    h = x / vega
    if order==1:
        update = h
    elif order==2:
        update = h * (1./(1 + 0.5*(h)*(vomma/vega)))
    elif order==3:
        update = (h 
                  * (1 + 0.5*(vomma/vega)*h)
                  / (1 + (vomma/vega)*h + (1./6.)*(ultima/vega)*h**2 ))
    else:
        raise ValueError("order must be 1,2 or 3, not {}".format(order))
    return update

def fwd_raw_compute_iv(tgt, fwd, strike, tau, is_call,
                       initial_sigma=2., num_iters=12, order=3):
    '''Apply Newton-like iteration to solve for implied vol with no error checks.
    
    '''
    sigma = initial_sigma * (1. + 0*tgt)
    for it in range(num_iters):
        greeks = fwd_greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma, is_call=is_call)
        update = refine_iv(tgt=tgt, 
                           price=greeks['price'], 
                           vega=greeks['vega'], 
                           vomma=greeks['vomma'], 
                           ultima=greeks['ultima'], 
                           order=order)
        sigma += update
    return sigma

def fwd_safe_compute_iv(tgt, fwd, strike, tau, is_call,
                        initial_sigma=1.5,
                        num_iters=12, 
                        order=3,
                        sigma_bounds=(0.01,2.),
                        price_tol=None):
    '''Apply Newton-like iteration to solve for implied vol with some error checking'''
    greeks_low = fwd_greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma_bounds[0], is_call=is_call)
    greeks_high = fwd_greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma_bounds[1], is_call=is_call)

    clip_tgt = np.clip(tgt, greeks_low['price'], greeks_high['price'])
    
    iv = fwd_raw_compute_iv(clip_tgt, fwd, strike, tau, is_call,
                            initial_sigma=initial_sigma,
                            num_iters=num_iters, order=order)

    greeks = fwd_greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=iv, is_call=is_call)
    #iv[clip_tgt != tgt] = np.nan # todo: float equality check is sometimes not what we want
    if price_tol is not None:
        iv[np.abs(greeks['price']-tgt)>price_tol] = np.nan
    
    return iv

def spot_compute_iv(tgt, spot, strike, tau, is_call, int_rate, div_yld):
    fwd = spot * np.exp(tau * (int_rate - div_yld))
    fwd_tgt = tgt * np.exp(tau * int_rate)
    return fwd_safe_compute_iv(tgt=fwd_tgt, fwd=fwd, strike=strike, tau=tau, is_call=is_call)

def spot_greeks_from(spot, strike, tau, sigma, is_call, int_rate, div_yld):
    fwd = spot * np.exp(tau * (int_rate - div_yld)) 
    disc_fact = np.exp(-tau * int_rate)
    fwd_greeks = fwd_greeks_from(fwd, strike, tau, sigma, is_call)
    return dict(
        price=disc_fact * fwd_greeks['price'],
        delta=np.exp(-tau * div_yld) * fwd_greeks['delta'],
        theta=fwd_greeks['theta'], # todo: this is not exactly right
        vega=fwd_greeks['vega'],
        wt_vega=fwd_greeks['wt_vega'],
        gamma=np.exp(-tau * div_yld) * fwd_greeks['gamma'],
    )


try:
    import scipy.misc
    from scipy.special import logsumexp
    scipy.misc.logsumexp = logsumexp # horible hack.  windows sucks.  not needed on linux/osx
    import cvxpy as cp
    cp.affine_prod=str # horible hack.  windows sucks

    import nelson_siegel_svensson as nss
    from nelson_siegel_svensson.calibrate import calibrate_ns_ols, errorfn_ns_ols
except:
    print('Failed to import cvxpy or nelson_siegel_svensson. Some functions may not work.')


def solve_marks(chain, S0, dte, obj_weight=0., int_rate=None, div_yld=None):
    """[summary]

    Args:
        chain (DataFrame): Indexed by strike prices, columns are multi-index
            with mid,spread at first level and CALL,PUT at second
        S0 (float): An approximate value of the spot price
        dte (int): Days to expiration for the option chain
        obj_weight (float, optional):  Mix between objective functions:
            0 -> sum of absolute deviation, 1 -> sum of squared deviation.
        int_rate ([float,None], optional): If specified, constrains the discount factor.
        div_yld ([float,None], optional): If specified, constrains the forward price

    Returns:
        dict: Keys include forward price (F), discount factor (D), int rate (r),
              and a dataframe (marks) with columns for call and put marks
              as well as the deviation from mid price as percentage of bid/ask spread.
    """
    tau = dte / 365.
    num_strikes = chain.shape[0]
    psi = cp.Variable(name='psi')
    delta = cp.Variable(name='delta')
    eps_call = cp.Variable(num_strikes, name='eps_call')
    eps_put = cp.Variable(num_strikes, name='eps_put')

    S = S0*(1-psi)
    D = 1-delta
    K = chain.index.to_numpy()

    m_call = chain.mid.CALL.to_numpy()
    m_put = chain.mid.PUT.to_numpy()

    spr_call = chain.spread.CALL.to_numpy()
    spr_put = chain.spread.PUT.to_numpy()

    C = m_call + cp.mul_elemwise(spr_call, eps_call)
    P = m_put + cp.mul_elemwise(spr_put, eps_put)

    objective = (
        obj_weight * (cp.sum_squares(eps_call) + cp.sum_squares(eps_put))
        + (1-obj_weight) * (cp.sum_entries(cp.abs(eps_call)) + cp.sum_entries(cp.abs(eps_put)))   
    )
    
    constraints = [C - P - (S - D*K) == 0]
    if int_rate is not None:
        constraints.append(D == np.exp(-int_rate*tau))
    if div_yld is not None:
        assert int_rate is not None
        constraints.append(S == S0*np.exp(-div_yld*tau))
        
    prob = cp.Problem(
        cp.Minimize(objective),
        constraints,
    )
    prob.solve()
    
    marks = pd.DataFrame(dict(eps_call=np.array(eps_call.value)[:,0],
                                  eps_put=np.array(eps_put.value)[:,0],
                                  mark_call=np.array(C.value)[:,0],
                                  mark_put=np.array(P.value)[:,0]
                               ),
                           index=chain.index)
    marks['diff_call'] = marks.eps_call * spr_call
    marks['diff_put'] = marks.eps_put * spr_put

    return dict(
        delta=delta.value,
        psi=psi.value,
        S = S.value,
        F = S.value / D.value,
        D = D.value,
        r = np.log(D.value)/(-tau),
        marks=marks,
    )




