using Godot;
using System;

public partial class TeamGoal : Area3D
{
    [Export] GameManager gameManager; 
    [Export] int team;
    
    public override void _Ready() {
        BodyEntered += OnBodyEntered;
    }

    private void OnBodyEntered(Node3D body) {
        if (body.IsInGroup("ball")) {
            gameManager.SCOREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE(team);
        }
    }
}
