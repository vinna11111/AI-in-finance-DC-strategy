// TickDumper.cs
using System;
using System.IO;
using cAlgo.API;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class TickDumper : Robot
    {
        [Parameter("Filename (Documents)", DefaultValue = "tick_data.csv")]
        public string Filename { get; set; }

        private string path;

        protected override void OnStart()
        {
            var docs = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            path = Path.Combine(docs, Filename);
            if (!File.Exists(path))
                File.WriteAllText(path, "utc_iso,bid,ask\n");
            Print("TickDumper writing to: " + path);
        }

        protected override void OnTick()
        {
            try
            {
                string line = $"{Server.Time:o},{Symbol.Bid:F8},{Symbol.Ask:F8}\n";
                File.AppendAllText(path, line);
            }
            catch (Exception ex)
            {
                Print("Tick write failed: " + ex.Message);
            }
        }

        protected override void OnStop()
        {
            Print("TickDumper stopped.");
        }
    }
}