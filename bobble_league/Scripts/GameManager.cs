using Godot;
using System.Collections.Generic;
using System;
using System.IO;
using System.Text.Json;

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
    [Export(PropertyHint.Range, "1,128,1")] public float AIvsAITimeScale = 8f;

    // a turn that never settles gets ended after this many game seconds.
    // the longest turn seen in phase 7 was 7.6 s, so 10 s ends only a stuck turn
    [Export] public float MaxTurnSeconds = 10f;

    // the "stopped" speed below which a body is treated as settled, so a turn can
    // end. a higher value ends ai_vs_ai turns sooner (worth it for training speed);
    // human games keep the slower, nicer-feeling value. applied at runtime from the
    // play mode, so the scene's own value is only a fallback.
    [Export] public float TrainingStopSpeed = 1.0f;
    [Export] public float HumanStopSpeed = 0.3f;
    float StopSpeed => CurrentPlayMode == PlayMode.AI_VS_AI ? TrainingStopSpeed : HumanStopSpeed;

    // arena is scaled 0.5: side walls at x = +-10, back of the nets at z = +-18.5
    [Export] public float FieldHalfX = 10f;
    [Export] public float FieldHalfZ = 18.5f;

    [Export] DataManager dataManager;
    [Export] TextureRect jumps_care;

    // the champion's Elo is shown here during HUMAN_VS_AI (hidden in ai_vs_ai).
    [Export] Label aiEloLabel;
    // the sidecar json holding the shown model's Elo, relative to res:// (the
    // bobble_league project folder). evaluate.py --human defaults to
    // best_final.pt, so the human HUD shows that model's score. read on _Ready
    // with GlobalizePath, so it works running the scene from the editor.
    [Export] public string AiEloSidecar = "../bobble_model/checkpoints/best_final.elo.json";

    int turn = 1;
    int currentTeam = 1;

    int scoreA;
    int scoreB;

    bool isSimulating;
    double simTime;
    float stepReward;
    bool currentDone;
    int lastScorer; // 0 nobody, 1 learner, 2 opponent
    bool turnHitOob; // this turn went out of bounds and was kicked back to centre

    // Phase 12, step 1: random-start layouts. seeded per process so parallel
    // Godots do not all scatter to the same board; python decides WHICH resets
    // are random (byte 91), Godot decides the positions.
    readonly RandomNumberGenerator rng = new RandomNumberGenerator();

    // the team the AI plays, python always sees the game from this team's side
    int LearnerTeam => CurrentPlayMode == PlayMode.HUMAN_VS_AI ? 2 : 1;
    int OpponentTeam => LearnerTeam == 1 ? 2 : 1;

    public override void _EnterTree() {
        Spawn();
    }

    public override void _Ready() {
        rng.Randomize();
        ReadCommandLine();
        ApplyStopSpeed();
        ShowAiElo();
        if (CurrentPlayMode == PlayMode.AI_VS_AI && AIvsAITimeScale > 1f) {
            SetTimeScale(AIvsAITimeScale);
        }
    }

    // show the AI's Elo in HUMAN_VS_AI, read from the sidecar json evaluate.py
    // writes next to the checkpoint. hidden in the other modes. a missing or bad
    // file shows "AI Elo unrated" rather than failing, matching evaluate.py.
    void ShowAiElo() {
        if (aiEloLabel == null) return;
        if (CurrentPlayMode != PlayMode.HUMAN_VS_AI) {
            aiEloLabel.Visible = false;
            return;
        }
        aiEloLabel.Text = "AI Elo " + ReadEloSidecar();
        aiEloLabel.Visible = true;
    }

    // "1104" from the sidecar's 'elo', or "unrated" when it is missing, unreadable
    // or has no usable number. the path is relative to res:// (the project folder).
    string ReadEloSidecar() {
        try {
            string path = ProjectSettings.GlobalizePath("res://").PathJoin(AiEloSidecar);
            if (!File.Exists(path)) return "unrated";
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
            if (doc.RootElement.TryGetProperty("elo", out JsonElement elo)
                && elo.TryGetDouble(out double value)) {
                return Mathf.RoundToInt(value).ToString();
            }
        }
        catch (Exception e) {
            GD.PrintErr($"Could not read AI Elo sidecar: {e.Message}");
        }
        return "unrated";
    }

    // push the mode's stop speed onto the ball and every current player. the ball
    // is set once; players are respawned each turn, so Spawn re-applies it to each
    void ApplyStopSpeed() {
        if (ball is Ball b) b.SetStopSpeed(StopSpeed);
        foreach (Node node in players.GetChildren()) {
            if (node is Player p) p.SetStopSpeed(StopSpeed);
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
        Reset(false);
    }

    // randomStart (Phase 12, step 1): scatter the ball and all six players to a
    // random legal non-overlapping layout instead of kickoff, so the AI also sees
    // odd boards (defending near its own goal, a scramble). A goal reset and an
    // out-of-bounds reset always use kickoff (randomStart false); only a
    // python-requested env reset with byte 91 set scatters.
    void Reset(bool randomStart) {
        foreach (Node node in players.GetChildren()) {
            node.Free();
        }
        ball.AngularVelocity = Vector3.Zero;
        ball.LinearVelocity = Vector3.Zero;
        if (randomStart) {
            SpawnRandom();
        } else {
            ball.Position = new Vector3(0, 0.5f, 0);
            Spawn();
        }
    }

    void Spawn() {
        foreach (PlayerSpawn spawn in spawns.GetChildren()) {
            Player player = PlayerPrefab.Instantiate<Player>();
            player.Position = new Vector3(spawn.Position.X, 1.001f, spawn.Position.Z);
            
            MeshInstance3D mesh = player.GetChild(0) as MeshInstance3D;
            if (spawn.Team == 1) mesh.SetSurfaceOverrideMaterial(0, TeamA);
            else if (spawn.Team == 2) mesh.SetSurfaceOverrideMaterial(0, TeamB);
            
            player.team = spawn.Team;
            player.SetStopSpeed(StopSpeed);
            
            players.AddChild(player);
        }
    }

    // random legal non-overlapping layout: the ball plus one player per spawn
    // (keeping each spawn's team), placed inside a margin of the pitch so nothing
    // starts in a wall or a goal, and no two bodies start closer than MinSpawnGap.
    void SpawnRandom() {
        List<Vector3> placed = new List<Vector3>();
        Vector3 ballXZ = SampleFreeSpot(placed);
        ball.Position = new Vector3(ballXZ.X, 0.5f, ballXZ.Z);
        placed.Add(ballXZ);

        foreach (PlayerSpawn spawn in spawns.GetChildren()) {
            Player player = PlayerPrefab.Instantiate<Player>();
            Vector3 spot = SampleFreeSpot(placed);
            player.Position = new Vector3(spot.X, 1.001f, spot.Z);
            placed.Add(spot);

            MeshInstance3D mesh = player.GetChild(0) as MeshInstance3D;
            if (spawn.Team == 1) mesh.SetSurfaceOverrideMaterial(0, TeamA);
            else if (spawn.Team == 2) mesh.SetSurfaceOverrideMaterial(0, TeamB);

            player.team = spawn.Team;
            player.SetStopSpeed(StopSpeed);

            players.AddChild(player);
        }
    }

    // bodies start inside this margin of the walls / goal lines (world units), so
    // a random layout never puts one in a wall or already in a goal mouth.
    const float SpawnMarginX = 1.5f;
    const float SpawnMarginZ = 2.5f;
    // smallest allowed centre-to-centre gap between two started bodies (radii are
    // ~0.5 each, so 2.0 leaves a clear gap and no start-of-turn overlap).
    const float MinSpawnGap = 2.0f;

    // rejection-sample a spot at least MinSpawnGap from every already-placed body.
    // after many tries it returns the best (farthest-from-neighbours) candidate so
    // a reset never hangs; with 7 bodies on this pitch a free spot is found fast.
    Vector3 SampleFreeSpot(List<Vector3> placed) {
        float xLimit = FieldHalfX - SpawnMarginX;
        float zLimit = FieldHalfZ - SpawnMarginZ;
        Vector3 best = Vector3.Zero;
        float bestMinDist = -1f;
        for (int attempt = 0; attempt < 100; attempt++) {
            Vector3 cand = new Vector3(
                rng.RandfRange(-xLimit, xLimit), 0f, rng.RandfRange(-zLimit, zLimit));
            float minDist = float.MaxValue;
            foreach (Vector3 p in placed) {
                float d = new Vector2(cand.X - p.X, cand.Z - p.Z).Length();
                if (d < minDist) minDist = d;
            }
            if (placed.Count == 0 || minDist >= MinSpawnGap) {
                return cand;
            }
            if (minDist > bestMinDist) {
                bestMinDist = minDist;
                best = cand;
            }
        }
        return best;
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
                turnHitOob = true;   // tell python to skip this turn's position shaping
                Reset();
            }
            bool timedOut = simTime > MaxTurnSeconds;
            if (timedOut) LogMovingBodies();
            if (timedOut || (StationaryBall() && StationaryPlayers())) {
                // only the goal reward (+1/-1/0) comes from here, python adds the shaping.
                // simTime is how long the turn took to settle in game seconds.
                dataManager.Export(currentDone, stepReward, false, LearnerTeam, lastScorer, (int)CurrentPlayMode, (float)simTime, turnHitOob);
                stepReward = 0;
                currentDone = false;
                lastScorer = 0;
                turnHitOob = false;
                isSimulating = false;
            }
            return;
        }

        if (CurrentPlayMode == PlayMode.HUMAN_VS_HUMAN) return; // no python here

        // handle resets before waiting on the human so env.reset() doesn't use up a right click
        if (dataManager.IsResetRequested()) {
            // v5: python may ask (byte 91) for a scattered random layout here
            Reset(dataManager.IsRandomStartRequested());
            stepReward = 0;
            currentDone = false;
            lastScorer = 0;
            turnHitOob = false;
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
            // wrap the angle (the aim axis is circular), clamp the power. this
            // matches python's wrap_angle: a=1.1 -> -0.9, a=-1.1 -> 0.9. PosMod
            // keeps the sign of the divisor, so negatives wrap the same way
            // python's % does (a plain C# % would leave them out of range).
            float a = Mathf.PosMod(action[2 * i] + 1f, 2f) - 1f;
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
            GD.Print($"  ball out at {ball.GlobalPosition} speed {ball.LinearVelocity.Length():0.0} vel {ball.LinearVelocity}");
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
