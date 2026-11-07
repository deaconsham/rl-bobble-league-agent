/*
Opt-in check (--settle-check) that a turn only ends once every body has stopped.
Prints one line per turn, read by tests/live/test_settle_live.py:
  SETTLE turn=N secs=S timed_out=0|1 stop_ball=V stop_player=V drift_ball=D drift_player=D wait_ticks=K
stop_* is the fastest speed a body had on the tick before it hit 0 (a body frozen
mid-roll shows up big). drift_* is how far a body moved after its position was sent.
*/

using Godot;
using System.Collections.Generic;

public class SettleCheck {
    readonly Dictionary<ulong, float> lastSpeed = new Dictionary<ulong, float>();
    readonly List<Vector3> sentAt = new List<Vector3>();
    float stopBall, stopPlayer, driftBall, driftPlayer;
    int waitTicks;
    string pending; // the line for the turn just sent, printed once the wait ends

    public static bool Requested() {
        return System.Array.IndexOf(OS.GetCmdlineUserArgs(), "--settle-check") >= 0;
    }

    // every physics tick while a turn plays
    public void OnSimTick(Node3D ball, List<Player> players) {
        Track(ball, HorizontalSpeed(ball), true);
        foreach (Player p in players) Track(p, HorizontalSpeed(p), false);
    }

    // a reset teleports bodies to rest, which is not a natural stop
    public void OnReset() {
        lastSpeed.Clear();
    }

    // the turn ended and its observation is being sent
    public void OnExport(Node3D ball, List<Player> players, int turn, double secs, bool timedOut) {
        Flush();
        sentAt.Clear();
        sentAt.Add(ball.GlobalPosition);
        foreach (Player p in players) sentAt.Add(p.GlobalPosition);
        pending = $"SETTLE turn={turn} secs={secs:0.00} timed_out={(timedOut ? 1 : 0)} " +
                  $"stop_ball={stopBall:0.000} stop_player={stopPlayer:0.000}";
        stopBall = stopPlayer = driftBall = driftPlayer = 0f;
        waitTicks = 0;
        lastSpeed.Clear();
    }

    // every physics tick while waiting for python's next request
    public void OnWaitTick(Node3D ball, List<Player> players) {
        if (pending == null || players.Count + 1 != sentAt.Count) return;
        waitTicks++;
        driftBall = Mathf.Max(driftBall, Flat(ball.GlobalPosition - sentAt[0]));
        for (int i = 0; i < players.Count; i++) {
            driftPlayer = Mathf.Max(driftPlayer, Flat(players[i].GlobalPosition - sentAt[i + 1]));
        }
    }

    // python asked for the next step or a reset, so the wait is over
    public void Flush() {
        if (pending == null) return;
        GD.Print($"{pending} drift_ball={driftBall:0.0000} drift_player={driftPlayer:0.0000} wait_ticks={waitTicks}");
        pending = null;
    }

    void Track(Node3D body, float speed, bool isBall) {
        lastSpeed.TryGetValue(body.GetInstanceId(), out float before);
        if (speed < 0.01f && before >= 0.01f) {
            if (isBall) stopBall = Mathf.Max(stopBall, before);
            else stopPlayer = Mathf.Max(stopPlayer, before);
        }
        lastSpeed[body.GetInstanceId()] = speed;
    }

    static float HorizontalSpeed(Node3D body) {
        return body is RigidBody3D rb ? Flat(rb.LinearVelocity) : 0f;
    }

    static float Flat(Vector3 v) {
        return new Vector2(v.X, v.Z).Length();
    }
}
