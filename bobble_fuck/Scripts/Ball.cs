using Godot;
using System;

public partial class Ball : RigidBody3D {
	[Export] float forfuckssake;
	public override void _PhysicsProcess(double delta) {
		if (new Vector2(LinearVelocity.X, LinearVelocity.Z).Length() < forfuckssake) {
			LinearVelocity = new Vector3(0, 0, 0);
			Position = new Vector3(Position.X, 0.5f, Position.Z);
			AngularVelocity = new Vector3(0, 0, 0);
		}
	}
}
