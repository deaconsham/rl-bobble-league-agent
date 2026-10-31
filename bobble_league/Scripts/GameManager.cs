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

    // Speeds up AI_VS_AI. Physics ticks are raised by the same factor, so each
    // physics step keeps the same dt and the simulation behaves the same, just faster.
    // Override from the command line: godot --path bobble_fuck -- --mode=ai_vs_ai --timescale=8
    [ExportGroup("Training speed")]
    [Export(PropertyHint.Range, "1,32,1")] public float AIvsAITimeScale = 8f;

    public override void _Ready() {
        ApplyCommandLine();
        if (CurrentPlayMode == PlayMode.AI_VS_AI && AIvsAITimeScale > 1f) {
            ApplyTimeScale(AIvsAITimeScale);
        }
    }

    // args after "--":  --mode=ai_vs_ai|human_vs_ai|human_vs_human  --timescale=N
    void ApplyCommandLine() {
        foreach (string arg in OS.GetCmdlineUserArgs()) {
            if (arg.StartsWith("--mode=")) {
                string mode = arg.Substring("--mode=".Length).ToUpperInvariant();
                if (Enum.TryParse(mode, out PlayMode parsed)) CurrentPlayMode = parsed;
                else GD.PrintErr($"Unknown --mode '{mode}'");
            }
            else if (arg.StartsWith("--timescale=")) {
                if (float.TryParse(arg.Substring("--timescale=".Length),
                        System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out float s) && s >= 1f) {
                    AIvsAITimeScale = s;
                }
                else GD.PrintErr($"Bad {arg}, need a number >= 1");
            }
        }
    }

    void ApplyTimeScale(float scale) {
        int baseTicks = Engine.PhysicsTicksPerSecond;
        Engine.TimeScale = scale;
        // Godot steps physics with dt = (1 / ticks) * time_scale, so scale ticks too
        Engine.PhysicsTicksPerSecond = Mathf.RoundToInt(baseTicks * scale);
        Engine.MaxPhysicsStepsPerFrame = Math.Max(8, Mathf.CeilToInt(scale) * 2);
        if (DisplayServer.GetName() != "headless") {
            DisplayServer.WindowSetVsyncMode(DisplayServer.VSyncMode.Disabled);
        }
        GD.Print($"AI_VS_AI time scale x{scale} ({Engine.PhysicsTicksPerSecond} physics ticks/s)");
    }

    int turn = 1;
    int currentTeam = 1;

    int scoreA;
    int scoreB;

    [Export] DataManager dataManager;
    bool isSimulating;
    float stepReward;
    bool currentDone;
    int lastScorer; // 0 none, 1 learner, 2 opponent

    // the team the RL policy plays. Python always sees this team's frame.
    int LearnerTeam => CurrentPlayMode == PlayMode.HUMAN_VS_AI ? 2 : 1;
    int OpponentTeam => LearnerTeam == 1 ? 2 : 1;

    public override void _Input(InputEvent @event) {
        if (@event is InputEventMouseButton mouseButton) {
            // H vs AI
            if (CurrentPlayMode == PlayMode.HUMAN_VS_AI) {
                if (humanWaitingForAI) return;
                bool stationaryBall = ball == null || ball.LinearVelocity.Length() < 0.01;
                if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.Pressed) {
                    if (stationaryBall && StationaryPlayers()) {
                        launchPlayer = playerClicked(1); // Humans can only click Team A
                        if (launchPlayer != null) {
                            launchPlayer.DrawArrow();
                        }
                    }
                }

                if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.IsReleased()) {
                    launchPlayer = null;
                }

                // Right click confirms human turn and passes to AI
                if (mouseButton.ButtonIndex == MouseButton.Right && mouseButton.IsReleased()) {
                    if (stationaryBall && StationaryPlayers()) {
                        // Turn over to AI
                        humanWaitingForAI = true;
                    }
                }
            }
            
            // H v H
            if (CurrentPlayMode == PlayMode.HUMAN_VS_HUMAN) {
                bool stationaryBall = StationaryBall();
                if (mouseButton.ButtonIndex == MouseButton.Left && mouseButton.Pressed) {
                    if (stationaryBall && StationaryPlayers()) {
                        launchPlayer = playerClicked(currentTeam);
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
                            // more poopy
                            foreach (Player player in players.GetChildren()) {
                                if (player.team == 1) {
                                    player.ClearArrow();
                                }
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
    }

    public override void _Process(double delta) {
        if (CurrentPlayMode is PlayMode.HUMAN_VS_HUMAN or PlayMode.HUMAN_VS_AI) {
            if (launchPlayer != null) {
                Vector3 launchVector = (new Vector3(launchPlayer.Position.X, 0, launchPlayer.Position.Z) - MousePosition()) / dragDistance;
                launchPlayer.input = launchVector;
            }
        }
    }

    // kind of shit but i didnt want to deal with godot raycast and colliders
    Player playerClicked(int desiredTeam) {
        // Compile the list of players
        // less poopy?
        List<Player> desiredPlayerList = new List<Player>();
        foreach (Player player in players.GetChildren()) {
            if (player.team == desiredTeam) {
                desiredPlayerList.Add(player);
            }
        }

        Vector3 mouse = MousePosition();
        
        // Closest player
        float closestDistance = float.MaxValue;
        Player closestPlayer = null;
        for (int i = 0; i < desiredPlayerList.Count; i++) {
            // faster calculation and doesnt matter lol idk ok cool
            // is a root calculation taht expensive ??
            float distance = desiredPlayerList[i].Position.DistanceSquaredTo(mouse);
            if (distance < closestDistance) {
                closestDistance = distance;
                closestPlayer = desiredPlayerList[i];
            }
        }

        if (closestDistance < selectDistance) {
            return closestPlayer;
        }
        else {
            return null;
        }
    }

    // Mouse position on xy plane
    Vector3 MousePosition() {
        Vector2 mousePosition = GetWindow().GetMousePosition();
        Plane plane = Plane.PlaneXZ;
        Vector3? intersect = plane.IntersectsRay(camera.Position, camera.ProjectRayNormal(mousePosition));
        if (intersect != null) {
            return intersect.Value;
        }
        else {
            return new Vector3(0, 0, 0);
        }
    }

    void Turn() {
        GD.Print("turn: " + turn);
        
        foreach (Player player in players.GetChildren()) {
            player.Launch();
        }
        
        turn++;
    }

    [Export] TextureRect jumps_care;
    
    public void SCOREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE(int team) {
        bool rlMode = CurrentPlayMode != PlayMode.HUMAN_VS_HUMAN;
        // a goal already ended this turn; ignore a second trigger
        if (rlMode && currentDone) return;

        if (team == 1) scoreA++;
        else scoreB++;

        if (rlMode) {
            // reward is from the LEARNER's side, whichever team that is
            bool learnerScored = team == LearnerTeam;
            stepReward += learnerScored ? 1.0f : -1.0f;
            lastScorer = learnerScored ? 1 : 2;
            currentDone = true; // tells Python the episode ended
        }

        GD.Print("team: " + team + " scored"); // soooo orangy
        Reset();
        jumps_care.SetVisible(true);
        if (jumps_care.Texture is AnimatedTexture animatedTexture) {
            animatedTexture.CurrentFrame = 0;
        }
    }

    public override void _EnterTree() {
        Spawn();
    }

    void Reset() {
        foreach (var node in players.GetChildren()) {
            var player = (Player)node; // yeah bro because casting would totally save us from an error bro again bro how many times do we have to tell you old man *farts*
            player.Free();
        }
        ball.AngularVelocity = Vector3.Zero;
        ball.LinearVelocity = Vector3.Zero;
        ball.Position = new Vector3(0, 0.5f, 0);
        Spawn();
    }
    
    bool StationaryPlayers() {
        bool value = true;
        foreach (Node node in players.GetChildren()) {
            if (node.IsQueuedForDeletion()) continue;
            if (node is Player p && p.LinearVelocity.Length() >= 0.01f) {
                value = false;
                break;
            }
        }
        return value;
    }

    // horizontal speed only, same rule as Ball.cs (it pins y, so y velocity is noise)
    bool StationaryBall() {
        return ball == null || new Vector2(ball.LinearVelocity.X, ball.LinearVelocity.Z).Length() < 0.01f;
    }

    void Spawn() {
        foreach (var node in spawns.GetChildren()) {
            var spawn = (PlayerSpawn)node;
            Player player = PlayerPrefab.Instantiate<Player>();
            player.Position = new Vector3(spawn.Position.X, 1.001f, spawn.Position.Z);
            
            MeshInstance3D mesh = player.GetChild(0) as MeshInstance3D;
            if (spawn.Team == 1) { mesh.SetSurfaceOverrideMaterial(0, TeamA);}
            else if (spawn.Team == 2) { mesh.SetSurfaceOverrideMaterial(0, TeamB);}
            
            player.team = spawn.Team;
            
            players.AddChild(player);
        }
        
        

        GD.Print(players.GetChildCount());
    }

    // can probably be a normal Process? idk man
    public override void _PhysicsProcess(double delta) {
        if (isSimulating) {
            simTime += delta;
            // a ball or player that leaves the pitch (e.g. kicked over a wall)
            // can never come to rest, so restart from kickoff. No reward, not done.
            if (!currentDone && AnythingOutOfBounds()) {
                GD.Print($"Turn {turn}: something left the pitch, back to kickoff");
                Reset();
            }
            bool timedOut = simTime > MaxTurnSeconds;
            if (timedOut) LogMovingBodies();
            if (timedOut || (StationaryBall() && StationaryPlayers())) {
                // Done simulating this turn. Reward here is only the goal
                // reward (+1/-1/0); Python adds reward shaping itself.
                dataManager.Export(currentDone, stepReward, false, LearnerTeam, lastScorer, (int)CurrentPlayMode);
                stepReward = 0;
                currentDone = false;
                lastScorer = 0;
                isSimulating = false;
            }
            return;
        }

        if (CurrentPlayMode == PlayMode.HUMAN_VS_HUMAN) return; // no Python here

        // Resets are handled before the human gate, so Python's first
        // env.reset() does not wait for (or eat) a human right-click.
        if (dataManager.IsResetRequested()) {
            Reset();
            stepReward = 0;
            currentDone = false;
            lastScorer = 0;
            humanWaitingForAI = false;
            dataManager.Export(false, 0f, true, LearnerTeam, 0, (int)CurrentPlayMode);
            return;
        }

        // For human vs ai mode, we don't want to consume Python's step request until the human has submitted their turn.
        if (CurrentPlayMode == PlayMode.HUMAN_VS_AI && !humanWaitingForAI) {
            return;
        }

        if (!dataManager.TryReadStep(out float[] action, out float[] opponentAction)) {
            return;
        }

        if (CurrentPlayMode == PlayMode.AI_VS_AI) {
            ApplyTeamAction(action, LearnerTeam);
            // no opponent action from Python -> opponent stands still
            ApplyTeamAction(opponentAction, OpponentTeam);
        }
        else { // HUMAN_VS_AI
            // Human's inputs (Team A) are already set in player.input from _Process
            ApplyTeamAction(action, LearnerTeam);
            humanWaitingForAI = false;
        }

        foreach (Player p in TeamPlayers(1)) p.Launch();
        foreach (Player p in TeamPlayers(2)) p.Launch();
        simTime = 0;
        isSimulating = true;
    }

    // a turn that never comes to rest is ended after this many game seconds
    [Export] public float MaxTurnSeconds = 20f;
    double simTime;

    // pitch size in world units (Arena mesh is scaled 0.5): walls at x = +-10,
    // back of the nets at z = +-18.5. 1 unit of slack before "out".
    [Export] public float FieldHalfX = 10f;
    [Export] public float FieldHalfZ = 18.5f;

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
        if (ball != null && ball.LinearVelocity.Length() >= 0.01f)
            GD.PrintErr($"  ball pos {ball.GlobalPosition} vel {ball.LinearVelocity}");
        foreach (Node node in players.GetChildren()) {
            if (node is Player p && p.LinearVelocity.Length() >= 0.01f)
                GD.PrintErr($"  team {p.team} player pos {p.GlobalPosition} vel {p.LinearVelocity}");
        }
    }

    // same order DataManager uses for observations, so action i <-> obs slot i
    List<Player> TeamPlayers(int team) {
        List<Player> list = new List<Player>();
        foreach (Node node in players.GetChildren()) {
            if (node.IsQueuedForDeletion()) continue;
            if (node is Player p && p.team == team) list.Add(p);
        }
        return list;
    }

    // action = [p1_x, p1_z, p2_x, p2_z, p3_x, p3_z] in that team's own frame
    // (see DataManager.FrameZ). null = no input this turn.
    void ApplyTeamAction(float[] action, int team) {
        List<Player> teamPlayers = TeamPlayers(team);
        float zSign = DataManager.FrameZ(team);
        for (int i = 0; i < teamPlayers.Count; i++) {
            if (action == null || 2 * i + 1 >= action.Length) {
                teamPlayers[i].input = Vector3.Zero;
                continue;
            }
            float x = action[2 * i];
            float z = action[2 * i + 1] * zSign;
            teamPlayers[i].input = new Vector3(x, 0, z);
        }
    }
}
