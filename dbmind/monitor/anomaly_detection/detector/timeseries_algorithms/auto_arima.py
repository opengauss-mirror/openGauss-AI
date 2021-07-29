/*
 * Copyright (c) 2020 Huawei Technologies Co.,Ltd.
 *
 * openGauss is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
 
 

import re
import time

import pandas as pd
import pmdarima as pm

from .model import AlgModel

date_format = "%Y-%m-%d %H:%M:%S"


class AutoArima(AlgModel):
    def __init__(self):
        AlgModel.__init__(self)
        self.model = None
        self.end_date = None

    def fit(self, timeseries):
        try:
            timeseries = pd.DataFrame(timeseries, columns=['ds', 'y'])
            timeseries['ds'] = timeseries['ds'].map(lambda x: time.strftime(date_format, time.localtime(x)))
            timeseries.set_index(['ds'], inplace=True)
            timeseries.index = pd.to_datetime(timeseries.index)
            self.end_date = timeseries.index[-1]
            self.model = pm.auto_arima(timeseries, seasonal=True)
            self.model.fit(timeseries)
        except Exception:
            raise

    def forecast(self, forecast_periods):
        forecast_period, forecast_freq = re.match(r'(\d+)?([WDHMS])', forecast_periods).groups()
        try:
            if forecast_period is None:
                forecast_period = 1
            else:
                forecast_period = int(forecast_period)
            forecast_date_range = pd.date_range(start=self.end_date,
                                                periods=forecast_period + 1,
                                                freq=forecast_freq,
                                                closed='right')
            forecast_date_range = forecast_date_range.map(lambda x: x.strftime(date_format)).values
            forecast_value = self.model.predict(forecast_period)
            return forecast_date_range, forecast_value
        except Exception:
            raise
