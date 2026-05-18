# LSTID Forecast — Idea Note

Forecasting LSTID occurrences from geomagnetic indices (Kp, SYM-H, AE/AL,
SME/SML), solar wind parameters, and F10.7, using regional catalogues over
one solar cycle. Currently: European sector. Possible extension:
American + Australian sectors.

## Why build multi-region catalogues (Europe + America + Australia)

1. **Larger, more balanced training set.** One sector ≈ hundreds of events,
   with strong MLT/seasonal imbalance. Three sectors roughly triple the
   sample and fill MLT/season combinations Europe alone cannot cover.

2. **Separating global drivers from local response.** Kp, SYM-H, AE, solar
   wind coupling are *global*. Simultaneous response in all sectors →
   driver-response link is real and globally scalable. Response in only one
   sector → driver alone is insufficient (local time, conductivity, neutral
   wind, hemispheric asymmetry matter). A single-region catalogue cannot
   disentangle this.

3. **MLT / longitude as a feature, not noise.** Model
   P(LSTID | drivers, MLT, season, hemisphere) instead of marginalising
   over local time. Sharpens conditional probabilities: drivers predict
   *when globally*, MLT predicts *where*.

4. **Hemispheric asymmetry (N vs S).** Australia provides the southern
   counterpart at comparable geomagnetic latitudes to Europe. Enables tests
   of symmetric vs asymmetric response (IMF By, season, conductivity).

5. **Stronger generalisation tests.** Out-of-sector validation (train
   Europe+America, test Australia) is a much harder test than within-region
   splits — distinguishes learned physics from regional climatology.

6. **Storm-time coverage.** Major storms are rare per cycle. Three sectors
   sample each storm at three MLTs, critical because LSTID
   generation/propagation depends on storm phase *and* local time of the
   auroral source vs. the observing sector.

## What it does NOT add

- Won't help if the bottleneck is driver quality (1-min vs 1-hr solar wind)
  rather than sample size.
- Won't help for purely European operational nowcasting if cross-region
  information is not used as a feature.

## Practical first step

Before committing to two more catalogues, do a **learning-curve analysis**
on the European one: fit a baseline forecast model and look at skill score
vs. training sample size.
- Skill still climbing → more data (more regions) will pay off.
- Skill plateaued → bottleneck is elsewhere (driver features, model
  architecture). Effort better spent there.

## The latitude / lag problem (and how to turn it into a feature)

Different geomagnetic latitudes of the three regional chains imply
**different delays** between a global driver (e.g. AE spike) and LSTID
detection: shorter for higher-magnetic-latitude chains, longer for lower
ones, plus N–S asymmetry because the southern auroral oval is more offset
from the geographic pole.

This is a **feature, not a bug**, if handled correctly:

1. **Propagation-adjusted lag.** For each catalogue compute
   `effective_lag = onset_time − driver_time − expected_travel_time`,
   using a nominal LSTID speed (≈500 m/s, or fit per event) and the chain's
   mean geomagnetic latitude. Residual lag becomes physically meaningful
   (source location, propagation conditions) instead of geometric.

2. **Magnetic latitude as an explicit predictor.** Use each chain's mean
   CGM latitude as a continuous input rather than a categorical
   Europe/America/Australia label. The model learns the lag–latitude
   relation directly and avoids treating sectors as interchangeable.

3. **Forecast-target choice.** Three reasonable options:
   - **Per-sector**: P(LSTID in sector S in [t, t+Δt] | drivers). Three
     models, or one with sector as input. Lag differences absorbed
     naturally.
   - **Global**: P(LSTID anywhere in [t, t+Δt]). Easier statistically,
     less useful operationally.
   - **Source-referenced**: P(auroral source launches an LSTID | drivers),
     with sector observations as noisy detectors. Cleanest physics;
     requires cross-sector event association.

4. **Hemispheric asymmetry.** Southern auroral oval offset means Australian
   chains at the same CGM latitude as European ones sit at different
   geographic latitudes / source-MLT geometry. Using **CGM latitude**
   (not geographic) handles most of this. For the south, consider
   SuperMAG hemispheric indices (SME/SML) or per-hemisphere AE rather than
   global AE/AL alone.

5. **Cross-sector event association as a science product.** Linking an
   LSTID in Europe at t₁ to one in America at t₂ following the same driver
   measures the global response time at two latitudes — a science
   deliverable on top of the forecast.

## Bottom line

The differing lag across regions is exactly the information that makes a
multi-region catalogue scientifically richer than a single one — it adds
the spatial dimension of the response. The risk is only if sectors are
treated as interchangeable replicates. Encode CGM latitude, adjust for
propagation, and the "problem" becomes leverage.
