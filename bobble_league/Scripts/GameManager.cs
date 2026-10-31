using Godot;
using System.Collections.Generic;
using System;

public partial class GameManager : Node
{
    [Export] Camera3D camera;
    [Export] Node players;
    [Export] RigidBody3D ball;
    [Export] float selectDistance = 5f;
    [Export] float dragDistance = 2f;
    
    Player launchPlayer;
    
    [Export] Node spawns;

    [Export] Material TeamA;
    [Export] Material TeamB;
    [Export] PackedScene PlayerPrefab;

    public enum PlayMode {
        AI_VS_AI,
        HUMAN_VS_AI,
        HUMAN_VS_HUMAN
    }

    [Export] public PlayMode CurrentPlayMode;
    bool humanWaitingForAI;

    // speeds up AI_VS_AI. physics ticks go up by the same amount so each step is still 1/60 s
    // command line: godot --path bobble_league -- --mode=ai_vs_ai --timescale=32
    [ExportGroup("Training speed")]
    [Export(PropertyHint.Range, "1,32,1")] public float AIvsAITimeScale = 8f;

    // a turn that never settles gets ended after this many game seconds
    [Export] public float MaxTurnSeconds = 20f;

    // arena is scaled 0.5: side walls at x = +-10, back of the nets at z = +-18.5
    [Export] public float FieldHalfX = 10f;
    [Export] public float FieldHalfZ = 18.5f;

    [Export] DataManager dataManager;
    [Export] TextureRect jumps_care;

    int turn = 1;
    int currentTeam = 1;

    int scoreA;
    int scoreB;

    bool isSimulating;
    double simTime;
    float stepReward;
    bool currentDone;
    int lastScorer; // 0 nobody, 1 learner, 2 opponent

    // the team the AI plays, python always sees the game from this team's side
    int LearnerTeam => CurrentPlayMode == PlayMode.HUMAN_VS_AI ? 2 : 1;
    int OpponentTeam => LearnerTeam == 1 ? 2 : 1;

    public override void _EnterTree() {
        Spawn();
    }

    public override void _Ready() {
        ReadCommandLine();
        if (CurrentPlayMode == PlayMode.AI_VS_AI && AIvsAITimeScale > 1f) {
            SetTimeScale(AIvsAITimeScale);
        }
    }

    // args after "--": --mode=ai_vs_ai|human_vs_ai|human_vs_human --timescale=N
    void ReadCommandLine() {
        foreach (string arg in OS.GetCmdlineUserArgs()) {
            if (arg.StartsWith("--mode=")) {
                string mode = arg.Substring("--mode=".Length).ToUpperInvariant();
                if (Enum.TryParse(mode, out PlayMode parsed)) {
                    CurrentPlayMode = parsed;
                }
                else {
                    GD.PrintErr($"Unknown --mode '{mode}'");
                }
            }
            else if (arg.StartsWith("--timescale=")) {
                string value = arg.Substring("--timescale=".Length);
                bool ok = float.TryParse(value, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float scale);
                if (ok && scale >= 1f) {
                    AIvsAITimeScale = scale;
                }
                else {
                    GD.PrintErr($"Bad {arg}, needs a number >= 1");
                }
            }
        }
    }

    void SetTimeScale(float scale) {
        int baseTicks = Engine.PhysicsTicksPerSecond;
        Engine.TimeScale = scale;
        // godot uses dt = time_scale / ticks, so raise ticks too or the steps get bigger
        Engine.PhysicsTicksPerSecond = Mathf.RoundToInt(baseTicks * scale);
        Engine.MaxPhysicsStepsPerFrame = Math.Max(8, Mathf.CeilToInt(scale) * 2);
        if (DisplayServer.GetName() != "headless") {
            DisplayServer.WindowSetVsyncMode(DisplayServer.VSyncMode.Disabled);
        }
        GD.Print($"AI_VS_AI time scale x{scale} ({Engine.PhysicsTicksPerSecond} physics ticks/s)");
    }

    public override void _Input(InputEvent @event) {
        if (@event is not InputEventMouseButton mouseButton) return;

        // human vs ai
        if (CurrentPlayMode == PlayMode.HUMAN_VS_AI) {
            if (humanWaitingForAI) return;
            bool stationaryBall = StationaryBall();
            if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.Pressed) {
                if (stationaryBall && StationaryPlayers()) {
                    launchPlayer = PlayerClicked(1); // humans can only pick team A
                    if (launchPlayer != null) {
                        launchPlayer.DrawArrow();
                    }
                }
            }

            if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.IsReleased()) {
                launchPlayer = null;
            }

            // right click ends the human's turn and hands it to the AI
            if (mouseButton.ButtonIndex == MouseButton.Right && mouseButton.IsReleased()) {
                if (stationaryBall && StationaryPlayers()) {
                    humanWaitingForAI = true;
                }
            }
        }

        // human vs human
        if (CurrentPlayMode == PlayMode.HUMAN_VS_HUMAN) {
            bool stationaryBall = StationaryBall();
            if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.Pressed) {
                if (stationaryBall && StationaryPlayers()) {
                    launchPlayer = PlayerClicked(currentTeam);
                    if (launchPlayer != null) {
                        launchPlayer.DrawArrow();
                    }
                }
            }
        
            if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.IsReleased()) {
                launchPlayer = null;
            }
        
            if (mouseButton.ButtonIndex == MouseButton.Right && mouseButton.IsReleased()) {
                GD.Print("team: " + currentTeam);
                if (currentTeam == 1) {
                    if (stationaryBall) {
                        // hide team A's arrows so team B can't see them
                        foreach (Player player in TeamPlayers(1)) {
                            player.ClearArrow();
                        }
                        currentTeam = 2;
                    }
                }
                else {
                    if (stationaryBall && StationaryPlayers()) {
                        Turn();
                        currentTeam = 1;
                    }
                }
            }
        }
    }

    public override void _Process(double delta) {
        if (CurrentPlayMode is PlayMode.HUMAN_VS_HUMAN or PlayMode.HUMAN_VS_AI) {
            if (launchPlayer != null) {
                Vector3 launchVector = (new Vector3(launchPlayer.Position.X, 0, launchPlayer.Position.Z) - MousePosition()) / dragDistance;
                launchPlayer.input = launchVector;
                // drag length is the power now: short drag = soft shot, clamped to [0.2, 1]
                launchPlayer.launchPower = Mathf.Clamp(launchVector.Length(), MinPower, 1f);
            }
        }
    }

    // closest player on that team to the mouse. simpler than raycasting the colliders
    Player PlayerClicked(int desiredTeam) {
        Vector3 mouse = MousePosition();
        
        float closestDistance = float.MaxValue;
        Player closestPlayer = null;
        foreach (Player player in TeamPlayers(desiredTeam)) {
            // squared distance is fine for comparing
            float distance = player.Position.DistanceSquaredTo(mouse);
            if (distance < closestDistance) {
                closestDistance = distance;
                closestPlayer = player;
            }
        }

        if (closestDistance < selectDistance) {
            return closestPlayer;
        }
        return null;
    }

    // mouse position on the xz plane
    Vector3 MousePosition() {
        Vector2 mousePosition = GetWindow().GetMousePosition();
        Plane plane = Plane.PlaneXZ;
        Vector3? intersect = plane.IntersectsRay(camera.Position, camera.ProjectRayNormal(mousePosition));
        if (intersect != null) {
            return intersect.Value;
        }
        return new Vector3(0, 0, 0);
    }

    void Turn() {
        GD.Print("turn: " + turn);
        
        foreach (Player player in players.GetChildren()) {
            player.Launch();
        }
        
        turn++;
    }

    // called by TeamGoal, team = the team that gets the point
    public void Score(int team) {
        bool rlMode = CurrentPlayMode != PlayMode.HUMAN_VS_HUMAN;
        // a goal already ended this turn, ignore the second trigger
        if (rlMode && currentDone) return;

        if (team == 1) scoreA++;
        else scoreB++;

        if (rlMode) {
            // reward is from the AI's side, whichever team that is
            bool learnerScored = team == LearnerTeam;
            stepReward += learnerScored ? 1.0f : -1.0f;
            lastScorer = learnerScored ? 1 : 2;
            currentDone = true; // ends the episode on the python side
        }

        GD.Print("team: " + team + " scored");
        Reset();
        jumps_care.SetVisible(true);
        if (jumps_care.Texture is AnimatedTexture animatedTexture) {
            animatedTexture.CurrentFrame = 0;
        }
    }

    void Reset() {
        foreach (Node node in players.GetChildren()) {
            node.Free();
        }
        ball.AngularVelocity = Vector3.Zero;
        ball.LinearVelocity = Vector3.Zero;
        ball.Position = new Vector3(0, 0.5f, 0);
        Spawn();
    }

    void Spawn() {
        foreach (PlayerSpawn spawn in spawns.GetChildren()) {
            Player player = PlayerPrefab.Instantiate<Player>();
            player.Position = new Vector3(spawn.Position.X, 1.001f, spawn.Position.Z);
            
            MeshInstance3D mesh = player.GetChild(0) as MeshInstance3D;
            if (spawn.Team == 1) mesh.SetSurfaceOverrideMaterial(0, TeamA);
            else if (spawn.Team == 2) mesh.SetSurfaceOverrideMaterial(0, TeamB);
            
            player.team = spawn.Team;
            
            players.AddChild(player);
        }
    }
    
    bool StationaryPlayers() {
        foreach (Node node in players.GetChildren()) {
            if (node.IsQueuedForDeletion()) continue;
            if (node is Player p && p.LinearVelocity.Length() >= 0.01f) {
                return false;
            }
        }
        return true;
    }

    // side to side speed only, same as Ball.cs (it pins y so y velocity is just noise)
    bool StationaryBall() {
        return ball == null || new Vector2(ball.LinearVelocity.X, ball.LinearVelocity.Z).Length() < 0.01f;
    }

    // same order DataManager uses, so action i goes to the player in obs slot i
    List<Player> TeamPlayers(int team) {
        List<Player> list = new List<Player>();
        foreach (Node node in players.GetChildren()) {
            if (node.IsQueuedForDeletion()) continue;
            if (node is Player p && p.team == team) list.Add(p);
        }
        return list;
    }

    public override void _PhysicsProcess(double delta) {
        if (isSimulating) {
            simTime += delta;
            // something that leaves the pitch can never settle, so go back to kickoff (no reward)
            if (!currentDone && AnythingOutOfBounds()) {
                GD.Print($"Turn {turn}: something left the pitch, back to kickoff");
                Reset();
            }
            bool timedOut = simTime > MaxTurnSeconds;
            if (timedOut) LogMovingBodies();
            if (timedOut || (StationaryBall() && StationaryPlayers())) {
                // only the goal reward (+1/-1/0) comes from here, python adds the shaping
                dataManager.Export(currentDone, stepReward, false, LearnerTeam, lastScorer, (int)CurrentPlayMode);
                stepReward = 0;
                currentDone = false;
                lastScorer = 0;
                isSimulating = false;
            }
            return;
        }

        if (CurrentPlayMode == PlayMode.HUMAN_VS_HUMAN) return; // no python here

        // handle resets before waiting on the human so env.reset() doesn't use up a right click
        if (dataManager.IsResetRequested()) {
            Reset();
            stepReward = 0;
            currentDone = false;
            lastScorer = 0;
            humanWaitingForAI = false;
            dataManager.Export(false, 0f, true, LearnerTeam, 0, (int)CurrentPlayMode);
            return;
        }

        // in human vs ai, leave python's step request alone until the human ends their turn
        if (CurrentPlayMode == PlayMode.HUMAN_VS_AI && !humanWaitingForAI) {
            return;
        }

        if (!dataManager.TryReadStep(out float[] action, out float[] opponentAction)) {
            return;
        }

        ApplyTeamAction(action, LearnerTeam);
        if (CurrentPlayMode == PlayMode.AI_VS_AI) {
            // null when python sends nothing, then the opponent stands still
            ApplyTeamAction(opponentAction, OpponentTeam);
        }
        else {
            // the human's moves are already in player.input from _Process
            humanWaitingForAI = false;
        }

        foreach (Player p in TeamPlayers(1)) p.Launch();
        foreach (Player p in TeamPlayers(2)) p.Launch();
        simTime = 0;
        isSimulating = true;
    }

    // action = [a1, p1, a2, p2, a3, p3] in that team's mirrored frame (protocol v3).
    // angle a -> theta = a*pi from -z (seam at own goal), power p -> 0.2+0.8*(p+1)/2.
    // matches bobble_league_env.decode_action. FrameZ flips z so team 2 aims the
    // same world direction team 1 would from the mirror image.
    const float MinPower = 0.2f;
    void ApplyTeamAction(float[] action, int team) {
        List<Player> teamPlayers = TeamPlayers(team);
        float zSign = DataManager.FrameZ(team);
        for (int i = 0; i < teamPlayers.Count; i++) {
            if (action == null || 2 * i + 1 >= action.Length) {
                teamPlayers[i].input = Vector3.Zero;
                teamPlayers[i].launchPower = 1f;
                continue;
            }
            float a = Mathf.Clamp(action[2 * i], -1f, 1f);
            float p = Mathf.Clamp(action[2 * i + 1], -1f, 1f);
            float theta = a * Mathf.Pi;
            // (x, z) unit aim: a=0 -> -z (enemy goal), a=+-1 -> +z (own goal)
            float x = Mathf.Sin(theta);
            float z = -Mathf.Cos(theta) * zSign;
            teamPlayers[i].input = new Vector3(x, 0, z);
            teamPlayers[i].launchPower = MinPower + (1f - MinPower) * (p + 1f) / 2f;
        }
    }

    // 1 unit of slack past the walls before it counts as out
    bool OutOfBounds(Vector3 p) {
        return Mathf.Abs(p.X) > FieldHalfX + 1f || Mathf.Abs(p.Z) > FieldHalfZ + 1f || p.Y < -2f;
    }

    bool AnythingOutOfBounds() {
        if (ball != null && OutOfBounds(ball.GlobalPosition)) {
            GD.Print($"  ball out at {ball.GlobalPosition}");
            return true;
        }
        foreach (Node node in players.GetChildren()) {
            if (node.IsQueuedForDeletion()) continue;
            if (node is Player p && OutOfBounds(p.GlobalPosition)) {
                GD.Print($"  {p.Name} (team {p.team}) out at {p.GlobalPosition}");
                return true;
            }
        }
        return false;
    }

    void LogMovingBodies() {
        GD.PrintErr($"Turn {turn} did not settle in {MaxTurnSeconds}s, ending it anyway. Still moving:");
        if (ball != null && ball.LinearVelocity.Length() >= 0.01f) {
            GD.PrintErr($"  ball pos {ball.GlobalPosition} vel {ball.LinearVelocity}");
        }
        foreach (Node node in players.GetChildren()) {
            if (node is Player p && p.LinearVelocity.Length() >= 0.01f) {
                GD.PrintErr($"  team {p.team} player pos {p.GlobalPosition} vel {p.LinearVelocity}");
            }
        }
    }
}
