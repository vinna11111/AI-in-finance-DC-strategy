// DC_AI_TimeseriesTrader.cs
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class DC_AI_TimeseriesTrader : Robot
    {
        [Parameter("Param filename (Documents)", DefaultValue = "theta_y_timeseries.csv")]
        public string ParamFilename { get; set; }

        [Parameter("Risk % of balance per trade", DefaultValue = 1.0)]
        public double RiskPercent { get; set; }

        [Parameter("Min signal prob to take entry (0 = ignore)", DefaultValue = 0.0)]
        public double MinSignalProb { get; set; }

        [Parameter("Enable trade CSV log (Documents)", DefaultValue = true)]
        public bool EnableTradeLog { get; set; }

        private string paramPath;
        private string tradeLogPath;
        private StreamWriter tradeLogWriter;

        private class ParamRow
        {
            public DateTime TimestampUtc;
            public double Theta;
            public double Y;
            public int Regime;
            public double AvgSignalProb;
        }
        private List<ParamRow> paramSeries = new List<ParamRow>();
        private int paramIndex = -1; // current pointer

        private enum Direction { Unknown = 0, Up = 1, Down = -1 }
        private Direction currentDir;
        private double referencePrice;
        private double entryPrice;
        private double maxExtrema;
        private Position myPosition;

        private double Theta = 0.001;
        private double Y = 0.5;
        private double CurrAvgSignalProb = 0.0;

        protected override void OnStart()
        {
            var docs = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            paramPath = Path.Combine(docs, ParamFilename);
            tradeLogPath = Path.Combine(docs, "dc_trade_log_timeseries.csv");

            LoadParamSeries();

            if (EnableTradeLog)
            {
                tradeLogWriter = new StreamWriter(tradeLogPath, true);
                if (new FileInfo(tradeLogPath).Length == 0)
                    tradeLogWriter.WriteLine("utc_iso,side,entry_price,exit_price,profit_pct,theta,Y,regime,avg_signal_prob");
                tradeLogWriter.Flush();
            }

            currentDir = Direction.Unknown;
            referencePrice = (Symbol.Bid + Symbol.Ask) / 2.0;

            Print($"DC_AI_TimeseriesTrader started, loaded param rows = {paramSeries.Count}");
        }

        protected override void OnStop()
        {
            tradeLogWriter?.Flush();
            tradeLogWriter?.Close();
            tradeLogWriter = null;
            Print("DC_AI_TimeseriesTrader stopped.");
        }

        protected override void OnTick()
        {
            DateTime now = Server.Time; // UTC
            UpdateParamPointer(now);

            double bid = Symbol.Bid;
            double ask = Symbol.Ask;
            double mid = (bid + ask) / 2.0;

            var existing = Positions.FindAll("DC_AI_TS");
            if (existing.Length == 0)
            {
                myPosition = null;
                DetectAndEnter(mid);
                return;
            }

            myPosition = existing.FirstOrDefault();
            if (myPosition == null) return;

            if (myPosition.TradeType == TradeType.Buy)
                ManageLong(bid, ask);
            else
                ManageShort(bid, ask);
        }

        private void LoadParamSeries()
        {
            paramSeries.Clear();
            try
            {
                var lines = File.ReadAllLines(paramPath);
                foreach (var ln in lines.Skip(1))
                {
                    if (string.IsNullOrWhiteSpace(ln)) continue;
                    var p = ln.Split(',');
                    if (p.Length < 3) continue;
                    DateTime ts;
                    if (!DateTime.TryParseExact(p[0].Trim(), "yyyy-MM-dd'T'HH:mm:ss'Z'", CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out ts))
                    {
                        if (!DateTime.TryParse(p[0].Trim(), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out ts))
                            continue;
                    }
                    double theta = double.TryParse(p[1], out var t) ? t : double.NaN;
                    double y = double.TryParse(p[2], out var yy) ? yy : double.NaN;
                    int regime = 0;
                    double avgp = 0.0;
                    if (p.Length >= 4) int.TryParse(p[3], out regime);
                    if (p.Length >= 5) double.TryParse(p[4], out avgp);

                    paramSeries.Add(new ParamRow { TimestampUtc = ts.ToUniversalTime(), Theta = theta, Y = y, Regime = regime, AvgSignalProb = avgp });
                }
                paramSeries = paramSeries.OrderBy(r => r.TimestampUtc).ToList();
            }
            catch (Exception ex)
            {
                Print("Failed to load param series: " + ex.Message);
            }
        }

        // move pointer forward until next timestamp > now
        private void UpdateParamPointer(DateTime now)
        {
            if (paramSeries.Count == 0) return;
            while (paramIndex + 1 < paramSeries.Count && paramSeries[paramIndex + 1].TimestampUtc <= now)
            {
                paramIndex++;
                var row = paramSeries[paramIndex];
                if (!double.IsNaN(row.Theta) && !double.IsNaN(row.Y))
                {
                    Theta = row.Theta;
                    Y = row.Y;
                    CurrAvgSignalProb = row.AvgSignalProb;
                    Print($"[PARAM UPDATE] time={row.TimestampUtc:o} Theta={Theta:E6} Y={Y:E6} regime={row.Regime} avg_prob={CurrAvgSignalProb:F3}");
                }
            }
        }

        private void DetectAndEnter(double mid)
        {
            // check DC up
            if (currentDir != Direction.Up && mid >= referencePrice * (1.0 + Theta))
            {
                // apply meta-label filter: require average predicted probability >= MinSignalProb (if user set it)
                if (MinSignalProb > 0.0 && CurrAvgSignalProb < MinSignalProb)
                {
                    // skip this entry and set reference to mid to continue detection later
                    referencePrice = mid;
                    currentDir = Direction.Unknown;
                    Print("[SKIP ENTRY] avg_prob below threshold: " + CurrAvgSignalProb.ToString("F4"));
                    return;
                }
                currentDir = Direction.Up;
                EnterLong();
            }
            else if (currentDir != Direction.Down && mid <= referencePrice * (1.0 - Theta))
            {
                if (MinSignalProb > 0.0 && CurrAvgSignalProb < MinSignalProb)
                {
                    referencePrice = mid;
                    currentDir = Direction.Unknown;
                    Print("[SKIP ENTRY SHORT] avg_prob below threshold: " + CurrAvgSignalProb.ToString("F4"));
                    return;
                }
                currentDir = Direction.Down;
                EnterShort();
            }
        }

        private void EnterLong()
        {
            double entryAsk = Symbol.Ask;
            double stopLossPrice = referencePrice;
            double stopLossPips = Math.Max(1.0, Math.Abs(entryAsk - stopLossPrice) / Symbol.PipSize);

            double volumeUnits;
            try { volumeUnits = Symbol.VolumeForProportionalRisk(ProportionalAmountType.Balance, RiskPercent, stopLossPips); }
            catch { volumeUnits = Symbol.VolumeForProportionalRisk(ProportionalAmountType.Balance, RiskPercent, stopLossPips, RoundingMode.Down); }
            volumeUnits = Symbol.NormalizeVolumeInUnits(volumeUnits, RoundingMode.Down);
            if (volumeUnits <= 0) volumeUnits = Symbol.VolumeInUnitsMin;

            var result = ExecuteMarketOrder(TradeType.Buy, SymbolName, volumeUnits, "DC_AI_TS", stopLossPips, null);
            if (result.IsSuccessful)
            {
                entryPrice = result.Position.EntryPrice;
                maxExtrema = entryPrice;
                Print("[ENTER LONG] entry=" + entryPrice.ToString("F8"));
            }
            else Print("[ENTER LONG FAILED] " + result.Error);
        }

        private void EnterShort()
        {
            double entryBid = Symbol.Bid;
            double stopLossPrice = referencePrice;
            double stopLossPips = Math.Max(1.0, Math.Abs(entryBid - stopLossPrice) / Symbol.PipSize);

            double volumeUnits;
            try { volumeUnits = Symbol.VolumeForProportionalRisk(ProportionalAmountType.Balance, RiskPercent, stopLossPips); }
            catch { volumeUnits = Symbol.VolumeForProportionalRisk(ProportionalAmountType.Balance, RiskPercent, stopLossPips, RoundingMode.Down); }
            volumeUnits = Symbol.NormalizeVolumeInUnits(volumeUnits, RoundingMode.Down);
            if (volumeUnits <= 0) volumeUnits = Symbol.VolumeInUnitsMin;

            var result = ExecuteMarketOrder(TradeType.Sell, SymbolName, volumeUnits, "DC_AI_TS", stopLossPips, null);
            if (result.IsSuccessful)
            {
                entryPrice = result.Position.EntryPrice;
                maxExtrema = entryPrice;
                Print("[ENTER SHORT] entry=" + entryPrice.ToString("F8"));
            }
            else Print("[ENTER SHORT FAILED] " + result.Error);
        }

        private void ManageLong(double bid, double ask)
        {
            if (bid > maxExtrema) maxExtrema = bid;
            double maxOSV = Math.Max(0.0, (maxExtrema - entryPrice) / entryPrice);
            double thetaPrime = Theta * Y * Math.Exp(-Math.Min(maxOSV, 10.0));
            if (bid <= maxExtrema * (1.0 - thetaPrime))
            {
                var pos = myPosition;
                if (pos != null)
                {
                    int posId = pos.Id;
                    var closeResult = ClosePosition(pos);
                    if (closeResult.IsSuccessful)
                    {
                        double exitPrice = GetExitPriceFromHistory(posId, TradeType.Buy, Symbol.Bid);
                        double profitPct = (exitPrice - entryPrice) / entryPrice * 100.0;
                        Print("[CLOSED LONG] exit=" + exitPrice.ToString("F8") + " profit%=" + profitPct.ToString("F4"));
                        LogTrade("LONG", entryPrice, exitPrice, profitPct);
                    }
                    else Print("[CLOSE LONG FAILED] " + closeResult.Error);
                }
                referencePrice = bid;
                currentDir = Direction.Down;
            }
        }

        private void ManageShort(double bid, double ask)
        {
            if (ask < maxExtrema) maxExtrema = ask;
            double maxOSV = Math.Max(0.0, (entryPrice - maxExtrema) / entryPrice);
            double thetaPrime = Theta * Y * Math.Exp(-Math.Min(maxOSV, 10.0));
            if (ask >= maxExtrema * (1.0 + thetaPrime))
            {
                var pos = myPosition;
                if (pos != null)
                {
                    int posId = pos.Id;
                    var closeResult = ClosePosition(pos);
                    if (closeResult.IsSuccessful)
                    {
                        double exitPrice = GetExitPriceFromHistory(posId, TradeType.Sell, Symbol.Ask);
                        double profitPct = (entryPrice - exitPrice) / entryPrice * 100.0;
                        Print("[CLOSED SHORT] exit=" + exitPrice.ToString("F8") + " profit%=" + profitPct.ToString("F4"));
                        LogTrade("SHORT", entryPrice, exitPrice, profitPct);
                    }
                    else Print("[CLOSE SHORT FAILED] " + closeResult.Error);
                }
                referencePrice = ask;
                currentDir = Direction.Up;
            }
        }

        private double GetExitPriceFromHistory(int positionId, TradeType tradeType, double fallbackPrice)
        {
            try
            {
                var trades = History.FindByPositionId(positionId);
                if (trades != null && trades.Length > 0)
                {
                    var last = trades.OrderBy(t => t.ClosingTime).Last();
                    return last.ClosingPrice;
                }
                else
                {
                    var lastByLabel = History.FindLast("DC_AI_TS", SymbolName, tradeType);
                    if (lastByLabel != null)
                        return lastByLabel.ClosingPrice;
                }
            }
            catch (Exception ex)
            {
                Print("History lookup failed: " + ex.Message);
            }
            return fallbackPrice;
        }

        private void LogTrade(string side, double entry, double exit, double profitPct)
        {
            if (EnableTradeLog && tradeLogWriter != null)
            {
                try
                {
                    int regime = (paramIndex >= 0 && paramIndex < paramSeries.Count) ? paramSeries[paramIndex].Regime : -1;
                    tradeLogWriter.WriteLine($"{DateTime.UtcNow:o},{side},{entry:F8},{exit:F8},{profitPct:F6},{Theta:E6},{Y:E6},{regime},{CurrAvgSignalProb:F6}");
                    tradeLogWriter.Flush();
                }
                catch (Exception ex)
                {
                    Print("Trade log write failed: " + ex.Message);
                }
            }
        }
    }
}