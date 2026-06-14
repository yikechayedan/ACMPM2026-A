# Question 3 Hybrid Dynamic Model

Data are sampled every 2 hours, so direct forecast horizons are 2, 4, 6, 8, 10, and 12 hours.
The workbook also contains an hourly 7:00-19:00 table, linearly interpolated between direct
2-hour-grid forecasts.
Historical treated-water NTU is used as an autoregressive state. Missing February 2026 NTU
values are filled chronologically by the 2-hour model before the 2-12 hour forecasts are made.
The physical layer uses a Gamma residence-time-distribution over the previous 24 hours, with
the mean residence time adjusted by clear-well level and raw-water flow. XGBoost models then
learn the residual between the physical baseline and each future NTU target.

Main output: `question3_answer.xlsx`.