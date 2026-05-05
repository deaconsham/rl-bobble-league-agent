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

    int turn = 1;
    int currentTeam = 1;

    int scoreA;
    int scoreB;

    [Export] DataManager dataManager;
    bool isSimulating;
    float stepReward;
    bool currentDone;

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
            if (player.team == currentTeam) {
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
        if (team == 1) {
            scoreA++;
            // Agent is Team A, so scoring gives +1 reward
            stepReward += 1.0f;
        }
        else {
            scoreB++;
            // Getting scored on is -1 reward
            stepReward -= 1.0f;
        }
        GD.Print("team: " + team + "scored"); // soooo orangy
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
            player.QueueFree();
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

    bool StationaryBall() {
        return ball == null || ball.LinearVelocity.Length() < 0.01f;
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
    }

    // can probably be a normal Process? idk man
    public override void _PhysicsProcess(double delta) {
        if (isSimulating) {
            if (StationaryBall() && StationaryPlayers()) {
                // Done simulating this turn
                float finalReward = (stepReward) + GetDistanceReward();
                
                dataManager.Export(currentDone, finalReward);
                stepReward = 0;
                currentDone = false;
                isSimulating = false;
            }
            return;
        }

        // For human vs ai mode, we don't want to consume Python's step request until the human has submitted their turn.
        if (CurrentPlayMode == PlayMode.HUMAN_VS_AI && !humanWaitingForAI) {
            return;
        }

        bool resetRequested;
        float[] action = dataManager.ProcessRequests(out resetRequested); // spams null exception for some reason

        if (resetRequested) {
            Reset();
            dataManager.Export();
            stepReward = 0;
            currentDone = false;
            humanWaitingForAI = false;
            return;
        }

        // Only process step requests if Python provides an action
        if (action != null) {
            if (CurrentPlayMode == PlayMode.AI_VS_AI) {
                ApplyRLAction(action);
                isSimulating = true;
            } 
            else if (CurrentPlayMode == PlayMode.HUMAN_VS_AI) {
                // Apply ai's predicted actions ONLY to Team B
                ApplyRLActionHalf(action, 2);
                
                // Human's inputs (Team A) are already set in the player.input variables from _Process then launch
                foreach (Node node in players.GetChildren()) {
                    if (node is Player p) p.Launch();
                }
                
                isSimulating = true;
                humanWaitingForAI = false;
            }
        }
    }

    float GetDistanceReward() {
        if (ball == null) return 0f;
        // From Team A POV: opponent goal is at Z=-33. 
        // Closer to opponent goal (Z = -33) gives positive reward.
        // Closer to own goal (Z = 33) gives negative reward.
        return -ball.Position.Z / 330.0f;
    }

    void ApplyRLAction(float[] action) {
        int actionIdx = 0;

        // Apply to Team A first, then Team B (which matches the spawn order array creation)
        foreach (Node node in players.GetChildren()) {
            if (node.IsQueuedForDeletion()) continue; // these safety checks are stupid, i want to know if something dumb like this happens
            if (node is Player player) {
                if (actionIdx < action.Length) { // genuinely what does this even fucking mean
                    float x = action[actionIdx];
                    float y = action[actionIdx + 1];
                    actionIdx += 2;
                    
                    player.input = new Vector3(x, 0, y);
                    player.Launch();
                }
            }
        }
    }

    void ApplyRLActionHalf(float[] action, int teamTarget) {
        int actionIdx = 0;
        foreach (Node node in players.GetChildren()) {
            if (node.IsQueuedForDeletion()) continue;
            if (node is Player player) {
                if (actionIdx < action.Length) {
                    float x = action[actionIdx];
                    float y = action[actionIdx + 1];
                    actionIdx += 2;
                    
                    if (player.team == teamTarget) {
                        player.input = new Vector3(x, 0, y);
                    }
                }
            }
        }
    }
}
